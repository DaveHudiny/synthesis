
import paynt.synthesizer.synthesizer_ar
import paynt.synthesizer.synthesizer_hybrid
import paynt.synthesizer.synthesizer_ar_storm
import paynt.synthesizer.synthesizer

import paynt.quotient.quotient
import paynt.quotient.pomdp
import paynt.utils.timer

from paynt.quotient.pomdp import PomdpQuotient
from paynt.quotient.storm_pomdp_control import StormPOMDPControl
from paynt.family.family import Family

from paynt.synthesizer.synthesizer_rl import SynthesizerRL

import stormpy

from threading import Thread
from queue import Queue
import time

import logging
logger = logging.getLogger(__name__)


class SynthesizerPomdp:

    # If true explore only the main family
    incomplete_exploration = False

    def __init__(self, quotient: PomdpQuotient, method: str, storm_control: StormPOMDPControl = None):
        self.quotient = quotient
        self.synthesizer = None
        self.method = method
        if method == "ar":
            self.synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR
        elif method == "hybrid":
            self.synthesizer = paynt.synthesizer.synthesizer_hybrid.SynthesizerHybrid
        self.total_iters = 0

        self.storm_control = storm_control
        if storm_control is not None:
            self.storm_control.quotient = self.quotient
            
            if quotient.specification.optimality.minimizing:
                self.storm_control.comparer = lambda x, y: x <= y
            else:
                self.storm_control.comparer = lambda x, y: x >= y


            self.storm_control.pomdp = self.quotient.pomdp
            self.storm_control.spec_formulas = self.quotient.specification.stormpy_formulae()
            self.synthesis_terminate = False
            # SAYNT only works with abstraction refinement
            self.synthesizer = paynt.synthesizer.synthesizer_ar_storm.SynthesizerARStorm
            if self.storm_control.iteration_timeout is not None:
                self.saynt_timer = paynt.utils.timer.Timer()
                self.synthesizer.saynt_timer = self.saynt_timer
                self.storm_control.saynt_timer = self.saynt_timer

    def synthesize(self, family: Family = None, print_stats=True):
        if family is None:
            family = self.quotient.family
        synthesizer = self.synthesizer(self.quotient)
        family.constraint_indices = self.quotient.family.constraint_indices
        assignment = synthesizer.synthesize(
            family, keep_optimum=True, print_stats=print_stats)
        iters_mdp = synthesizer.stat.iterations_mdp if synthesizer.stat.iterations_mdp is not None else 0
        self.total_iters += iters_mdp
        return assignment

    def unfold_and_synthesize(self, mem_size, unfold_storm, unfold_imperfect_only=True):

        paynt.quotient.pomdp.PomdpQuotient.current_family_index = mem_size
        # unfold memory according to the best result
        if not unfold_storm:
            logger.info(
                "Synthesizing optimal k={} controller ...".format(mem_size))
            if unfold_imperfect_only:
                self.quotient.set_imperfect_memory_size(mem_size)
            else:
                self.quotient.set_global_memory_size(mem_size)
        else:
            if mem_size > 1:
                obs_memory_dict = {}
                if self.storm_control.is_storm_better:
                    # Storm's result is better and it needs memory
                    if self.storm_control.is_memory_needed():
                        obs_memory_dict = self.storm_control.memory_vector
                        logger.info(f'Added memory nodes to match Storm data')
                    else:
                        if self.storm_control.unfold_cutoff:
                            # consider the cut-off schedulers actions when updating memory
                            result_dict = self.storm_control.result_dict
                        else:
                            # only consider the induced DTMC without cut-off states
                            result_dict = self.storm_control.result_dict_no_cutoffs
                        for obs in range(self.quotient.observations):
                            if obs in result_dict:
                                obs_memory_dict[obs] = self.quotient.observation_memory_size[obs] + 1
                            else:
                                obs_memory_dict[obs] = self.quotient.observation_memory_size[obs]
                        logger.info(
                            f'Added memory nodes for observations based on Storm data')
                else:
                    for obs in range(self.quotient.observations):
                        if self.quotient.observation_states[obs] > 1:
                            obs_memory_dict[obs] = self.quotient.observation_memory_size[obs] + 1
                        else:
                            obs_memory_dict[obs] = 1
                    logger.info(
                        f'Increased memory in all imperfect observation')
                self.quotient.set_memory_from_dict(obs_memory_dict)

        family = self.quotient.family

        # if Storm's result is better, use it to obtain main family that considers only the important actions
        if self.storm_control.is_storm_better:
            if self.storm_control.use_cutoffs:
                # consider the cut-off schedulers actions
                result_dict = self.storm_control.result_dict
            else:
                # only consider the induced DTMC actions without cut-off states
                result_dict = self.storm_control.result_dict_no_cutoffs
            main_family = self.storm_control.get_main_restricted_family(
                family, result_dict)
            subfamily_restrictions = []
            if not self.storm_control.incomplete_exploration:
                subfamily_restrictions = self.storm_control.get_subfamilies_restrictions(
                    family, result_dict)
            subfamilies = self.storm_control.get_subfamilies(
                subfamily_restrictions, family)
        # if PAYNT is better continue normally
        else:
            main_family = family
            subfamilies = []

        self.synthesizer.subfamilies_buffer = subfamilies
        self.synthesizer.main_family = main_family
        assignment = self.synthesize(family)
        return assignment

    # iterative strategy using Storm analysis to enhance the synthesis
    def strategy_iterative_storm(self, unfold_imperfect_only, unfold_storm=True):
        '''
        @param unfold_imperfect_only if True, only imperfect observations will be unfolded
        '''
        mem_size = paynt.quotient.pomdp.PomdpQuotient.initial_memory_size
        self.synthesizer.storm_control = self.storm_control

        while True:
            assignment = self.unfold_and_synthesize(mem_size, unfold_storm)
            if assignment is not None:
                self.storm_control.latest_paynt_result = assignment
                self.storm_control.paynt_export = self.quotient.extract_policy(
                    assignment)
                self.storm_control.paynt_bounds = self.quotient.specification.optimality.optimum
                self.storm_control.paynt_fsc_size = self.quotient.policy_size(
                    self.storm_control.latest_paynt_result)
                self.storm_control.latest_paynt_result_fsc = self.quotient.assignment_to_fsc(
                    self.storm_control.latest_paynt_result)
            self.storm_control.update_data()

            if self.synthesis_terminate:
                break

            mem_size += 1

    def print_synthesized_controllers(self):
        hline = "\n------------------------------------\n"
        print(hline)
        print("PAYNT results: ")
        print(self.storm_control.paynt_bounds)
        print("controller size: {}".format(self.storm_control.paynt_fsc_size))
        print()
        print("Storm results: ")
        print(self.storm_control.storm_bounds)
        print("controller size: {}".format(
            self.storm_control.belief_controller_size))
        print(hline)

    def assign_rl_result(self, assignment):
        self.storm_control.latest_rl_result = assignment
        self.storm_control.rl_export = self.second_quotient.extract_policy(
            self.storm_control.latest_rl_result)
        self.storm_control.rl_bounds = self.second_quotient.specification.optimality.optimum
        self.storm_control.rl_fsc_size = self.second_quotient.policy_size(
            self.storm_control.latest_rl_result)
        self.storm_control.latest_rl_result_fsc = self.second_quotient.assignment_to_fsc(
            self.storm_control.latest_rl_result)
        self.storm_control.second_quotient = self.second_quotient
        self.storm_control.update_data()
        if self.storm_control.is_rl_better:
            self.storm_control.paynt_export = self.storm_control.rl_export

    def is_rl_better(self):
        if self.storm_control.rl_bounds is None:
            return False
        elif self.storm_control.paynt_bounds is None:
            return True
        
        if self.storm_control.rl_bounds < self.storm_control.paynt_bounds and self.quotient.specification.optimality.minimizing:
            return True
        elif self.storm_control.rl_bounds > self.storm_control.paynt_bounds and not self.quotient.specification.optimality.minimizing:
            return True
        return False
    
    def get_better_fsc(self):
        if self.is_rl_better():
            return self.storm_control.latest_rl_result_fsc
        return self.storm_control.latest_paynt_result_fsc
    
    def get_better_fsc_rl_less(self):
        if self.storm_control.latest_paynt_result_fsc is None and self.storm_control.latest_storm_fsc is None:
            return None
        elif self.storm_control.paynt_bounds is None:
            return self.storm_control.latest_storm_fsc
        elif self.storm_control.storm_bounds is None:
            return self.storm_control.latest_paynt_result_fsc
        elif self.storm_control.paynt_bounds <= self.storm_control.storm_bounds and self.quotient.specification.optimality.minimizing:
            return self.storm_control.latest_paynt_result_fsc
        elif self.storm_control.paynt_bounds >= self.storm_control.storm_bounds and not self.quotient.specification.optimality.minimizing:
            return self.storm_control.latest_paynt_result_fsc
        else:
            return self.storm_control.latest_storm_fsc

    def iterative_storm_loop(self, timeout, paynt_timeout, storm_timeout, iteration_limit=0):
        ''' Main SAYNT loop. '''

        self.interactive_queue = Queue()
        self.synthesizer.s_queue = self.interactive_queue
        self.storm_control.interactive_storm_setup()
        if hasattr(self, "input_rl_settings_dict"):
            rl_synthesizer = SynthesizerRL(
                self.second_quotient, self.method, self.storm_control, self.input_rl_settings_dict, True)
            self.loop = self.input_rl_settings_dict["loop"]
            self.rl_oracle = True
            agent_wrapper = rl_synthesizer.get_agents_wrapper()
        else:
            self.rl_oracle = False
        iteration = 1
        paynt_thread = Thread(target=self.strategy_iterative_storm, args=(
            True, self.storm_control.unfold_storm))

        iteration_timeout = time.time() + timeout
        timeout_bonus = 0
        self.saynt_timer.start()
        skip_first = False
        while True:
            if iteration == 1:
                paynt_thread.start()
            else:
                self.interactive_queue.put("resume")

            logger.info("Timeout for PAYNT started")
            time.sleep(paynt_timeout)
            self.interactive_queue.put("timeout")

            while not self.interactive_queue.empty():
                time.sleep(0.1)

            pre_rl_time = time.time()
            if not skip_first and self.rl_oracle and (self.loop or iteration == 1):
                fsc = self.get_better_fsc_rl_less()
                assignment = rl_synthesizer.single_shot_synthesis(agent_wrapper, nr_rl_iterations=1001,
                                                                  paynt_timeout=paynt_timeout,
                                                                  fsc=fsc)
                if assignment is not None:
                    print("RL assignment found")
                    self.assign_rl_result(assignment)
            skip_first = False

            timeout_bonus += time.time() - pre_rl_time

            if iteration == 1:
                self.storm_control.interactive_storm_start(storm_timeout)
            else:
                self.storm_control.interactive_storm_resume(storm_timeout)

            # compute sizes of controllers
            assert self.storm_control.latest_storm_result is not None, "Storm result is None"
            self.storm_control.belief_controller_size = self.storm_control.get_belief_controller_size(
                self.storm_control.latest_storm_result, self.storm_control.paynt_fsc_size)

            if time.time() > iteration_timeout + timeout_bonus or iteration == iteration_limit:
                break
            
            iteration += 1


        self.interactive_queue.put("terminate")
        self.synthesis_terminate = True
        paynt_thread.join()

        self.storm_control.interactive_storm_terminate()


        self.saynt_timer.stop()

        if self.storm_control.latest_storm_fsc is not None:
            # pre_clone_time = time.time()
            # start_time = time.time()
            # dtmc = self.quotient.get_induced_dtmc_from_fsc(self.storm_control.latest_storm_fsc)
            # result = stormpy.model_checking(dtmc, self.quotient.specification.optimality.formula)
            # print(result.at(0))
            # start_time = time.time()
            dtmc_vec = self.quotient.get_induced_dtmc_from_fsc_vec(self.storm_control.latest_storm_fsc)
            result_vec = stormpy.model_checking(dtmc_vec, self.quotient.specification.optimality.formula)
            print(result_vec.at(0))

    # run PAYNT POMDP synthesis with a given timeout
    def run_synthesis_timeout(self, timeout):
        self.interactive_queue = Queue()
        self.synthesizer.s_queue = self.interactive_queue
        paynt_thread = Thread(
            target=self.strategy_iterative_storm, args=(True, False))
        iteration_timeout = time.time() + timeout
        paynt_thread.start()

        while True:
            if time.time() > iteration_timeout:
                break

            time.sleep(1)

        self.interactive_queue.put("pause")
        self.interactive_queue.put("terminate")
        self.synthesis_terminate = True
        paynt_thread.join()

    # PAYNT POMDP synthesis that uses pre-computed results from Storm as guide

    def strategy_storm(self, unfold_imperfect_only, unfold_storm=True):
        '''
        @param unfold_imperfect_only if True, only imperfect observations will be unfolded
        '''
        mem_size = paynt.quotient.pomdp.PomdpQuotient.initial_memory_size
        self.synthesizer.storm_control = self.storm_control
        while True:
            if self.storm_control.is_storm_better == False:
                self.storm_control.parse_results(self.quotient)
            assignment = self.unfold_and_synthesize(
                mem_size, unfold_storm, unfold_imperfect_only)
            if assignment is not None:
                self.storm_control.latest_paynt_result = assignment
                self.storm_control.paynt_export = self.quotient.extract_policy(
                    assignment)
                self.storm_control.paynt_bounds = self.quotient.specification.optimality.optimum
            self.storm_control.update_data()
            mem_size += 1

    def strategy_iterative(self, unfold_imperfect_only):
        '''
        @param unfold_imperfect_only if True, only imperfect observations will be unfolded
        '''
        mem_size = paynt.quotient.pomdp.PomdpQuotient.initial_memory_size
        opt = self.quotient.specification.optimality.optimum
        while True:

            logger.info(
                "Synthesizing optimal k={} controller ...".format(mem_size))
            if unfold_imperfect_only:
                self.quotient.set_imperfect_memory_size(mem_size)
            else:
                self.quotient.set_global_memory_size(mem_size)

            self.synthesize(self.quotient.family)

            opt_old = opt
            opt = self.quotient.specification.optimality.optimum

            # finish if optimum has not been improved
            # if opt_old == opt and opt is not None:
            #     break
            mem_size += 1

            # break

    def set_reinforcement_learning(self, input_rl_settings_dict: dict):
        """ Set RL settings from PAYNT cli.
        Args:
            input_rl_settings_dict (dict): Dictionary with RL settings from PAYNT cli.
        """
        self.input_rl_settings_dict = input_rl_settings_dict

    def set_second_quotient(self, second_quotient: paynt.quotient.quotient.Quotient):
        """ Set second quotient for RL synthesis.
        Args:
            second_quotient (Quotient): Second quotient for RL synthesis.
        """
        self.second_quotient = second_quotient

    def export_fsc(self, export_filename_base):

        fsc_json = None
        if self.storm_control.saynt_fsc is not None:
            fsc_json = self.storm_control.saynt_fsc.__str__()
        elif self.storm_control.latest_paynt_result_fsc is not None:
            fsc_json = self.storm_control.latest_paynt_result_fsc.__str__()
        else:
            # TODO add export option for pure PAYNT synthesis
            pass
        
        assert fsc_json is not None, "No FSC to export"

        with open(export_filename_base + ".fsc.json", "w") as f:
            f.write(fsc_json)

        logger.info(f"Exported FSC to {export_filename_base}.fsc.json")

    def run(self, optimum_threshold=None):
        if self.storm_control.iteration_timeout is None and hasattr(self, "input_rl_settings_dict"):

            synthesizer_rl = SynthesizerRL(
                self.quotient, self.method, self.storm_control, self.input_rl_settings_dict)
            result = synthesizer_rl.run()
            return result

        if self.storm_control is None:
            # Pure PAYNT POMDP synthesis
            self.strategy_iterative(unfold_imperfect_only=True)
            return

        # SAYNT
        logger.info("Storm POMDP option enabled")
        logger.info("Storm settings: iterative - {}, get_storm_result - {}, storm_options - {}, prune_storm - {}, unfold_strategy - {}, use_storm_cutoffs - {}".format(
                    (self.storm_control.iteration_timeout, self.storm_control.paynt_timeout,
                     self.storm_control.storm_timeout), self.storm_control.get_result,
                    self.storm_control.storm_options, self.storm_control.incomplete_exploration, (
                        self.storm_control.unfold_storm, self.storm_control.unfold_cutoff), self.storm_control.use_cutoffs
                    ))
        # start SAYNT
        if self.storm_control.iteration_timeout is not None:
            self.iterative_storm_loop(timeout=self.storm_control.iteration_timeout,
                                      paynt_timeout=self.storm_control.paynt_timeout,
                                      storm_timeout=self.storm_control.storm_timeout,
                                      iteration_limit=0)
        # run PAYNT for a time given by 'self.storm_control.get_result' and then run Storm using the best computed FSC at cut-offs
        elif self.storm_control.get_result is not None:
            if self.storm_control.get_result:
                self.run_synthesis_timeout(self.storm_control.get_result)
            self.storm_control.run_storm_analysis()
        # run Storm and then use the obtained result to enhance PAYNT synthesis
        else:
            self.storm_control.get_storm_result()
            self.strategy_storm(unfold_imperfect_only=True,
                                unfold_storm=self.storm_control.unfold_storm)

        self.print_synthesized_controllers()

        if paynt.synthesizer.synthesizer.Synthesizer.export_synthesis_filename_base is not None:
            self.export_fsc(paynt.synthesizer.synthesizer.Synthesizer.export_synthesis_filename_base)
