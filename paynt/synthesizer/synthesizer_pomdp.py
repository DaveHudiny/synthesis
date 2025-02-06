import json
import numpy as np
from time import sleep
import os
from paynt.rl_extension.family_extractors.external_family_wrapper import ExtractedFamilyWrapper
from rl_src.experimental_interface import ArgsEmulator
import pickle
import stormpy

from .statistic import Statistic
import paynt.synthesizer.synthesizer_ar
from .synthesizer_ar_storm import SynthesizerARStorm
from .synthesizer_hybrid import SynthesizerHybrid
from .synthesizer_multicore_ar import SynthesizerMultiCoreAR
from ..rl_extension.saynt_rl_tools.agents_wrapper import AgentsWrapper

from paynt.rl_extension.saynt_rl_tools.q_values_correction import make_qvalues_table_tensorable
from ..rl_extension.saynt_rl_tools.rl_saynt_combo_modes import RL_SAYNT_Combo_Modes, init_rl_args

from paynt.rl_extension.saynt_rl_tools.regex_patterns import RegexPatterns
from paynt.parser.prism_parser import PrismParser
import paynt.synthesizer.synthesizer_hybrid
import paynt.synthesizer.synthesizer_ar_storm

import paynt.quotient.quotient
import paynt.quotient.pomdp
import paynt.utils.timer

import paynt.verification.property

from threading import Thread
from queue import Queue
import time

import logging
logger = logging.getLogger(__name__)

from rl_src.interpreters.fsc_based_interpreter import NaiveFSCPolicyExtraction
from rl_src.tools.evaluators import evaluate_extracted_fsc

from paynt.synthesizer.synthesizer_rl_storm_paynt import SynthesizerRL


class SynthesizerPomdp:

    # If true explore only the main family
    incomplete_exploration = False

    def __init__(self, quotient, method, storm_control):
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
            self.storm_control.pomdp = self.quotient.pomdp
            self.storm_control.spec_formulas = self.quotient.specification.stormpy_formulae()
            self.synthesis_terminate = False
            self.synthesizer = paynt.synthesizer.synthesizer_ar_storm.SynthesizerARStorm # SAYNT only works with abstraction refinement
            if self.storm_control.iteration_timeout is not None:
                self.saynt_timer = paynt.utils.timer.Timer()
                self.synthesizer.saynt_timer = self.saynt_timer
                self.storm_control.saynt_timer = self.saynt_timer

    def synthesize(self, family=None, print_stats=True, timer=None):
        if family is None:
            family = self.quotient.family
        synthesizer = self.synthesizer(self.quotient)
        family.constraint_indices = self.quotient.family.constraint_indices
        assignment = synthesizer.synthesize(
            family, keep_optimum=True, print_stats=print_stats, timeout=timer)
        iters_mdp = synthesizer.stat.iterations_mdp if synthesizer.stat.iterations_mdp is not None else 0
        self.total_iters += iters_mdp
        return assignment

    def get_qvalues_by_property(self, property: str = f"", prism=None, assignment=None):
        parsed_property = stormpy.parse_properties(property, context=prism)[0]
        opt_property = paynt.verification.property.OptimalityProperty(
            parsed_property)
        qvalues = self.quotient.compute_qvalues(assignment, prop=opt_property)
        tensorable_qvalues = make_qvalues_table_tensorable(qvalues)
        return tensorable_qvalues

    def fix_qvalues(self, assignment, original_qvalues, original_property_str):
        original_qvalues = make_qvalues_table_tensorable(original_qvalues)
        if "Pmax" in original_property_str:
            qvalues = np.multiply(
                original_qvalues, self.args.evaluation_goal)
            # TODO: More possible names of reward model
            try:
                reward_model_name = list(
                    self.quotient.pomdp.reward_models.keys())[-1]
            except:
                reward_model_name = "steps"
            prism = PrismParser.prism  # Not a good way to access the prism object
            trap_qvalues = self.get_qvalues_by_property(
                f"Pmin=? [ F (!\"notbad\" & !\"goal\")]", prism, assignment)
            cum_reward_qvalues = self.get_qvalues_by_property(
                f"R{{\"{reward_model_name}\"}}min=? [ C<={self.args.max_steps} ]", prism, assignment)
            qvalues = qvalues + trap_qvalues * \
                self.args.evaluation_antigoal - cum_reward_qvalues
        elif "Pmin" in original_property_str:
            # TODO: Make proper Pmin correction
            qvalues = self.args.evaluation_antigoal * original_qvalues
        elif RegexPatterns.check_max_property(original_property_str):
            qvalues = original_qvalues + self.args.evaluation_goal
        elif RegexPatterns.check_min_property(original_property_str):
            qvalues = (-original_qvalues) + self.args.evaluation_goal
        else:
            logger.info(
                f"Unknown property type: {original_property_str}. Using qvalues computed with given original property")
            qvalues = original_qvalues
        if self.args.normalize_simulator_rewards:
            qvalues = qvalues / self.args.evaluation_goal
        return qvalues

    def compute_qvalues_for_rl(self, assignment):
        original_property = self.quotient.get_property()
        original_property_qvalues = self.quotient.compute_qvalues(
            assignment, prop=original_property)
        original_property_str = original_property.__str__()
        qvalues = self.fix_qvalues(
            assignment, original_property_qvalues, original_property_str)
        return qvalues


    def unfold_and_synthesize(self, mem_size, unfold_storm):
        paynt.quotient.pomdp.PomdpQuotient.current_family_index = mem_size

        # unfold memory according to the best result
        if not unfold_storm:
            logger.info("Synthesizing optimal k={} controller ...".format(mem_size) )
            if unfold_imperfect_only:
                self.quotient.set_imperfect_memory_size(mem_size)
            else:
                self.quotient.set_global_memory_size(mem_size)
            return

        if mem_size <= 1:
            return
        obs_memory_dict = {}
        if self.storm_control.is_storm_better:
            # Storm's result is better and it needs memory
            if self.storm_control.is_memory_needed():
                obs_memory_dict = self.storm_control.memory_vector
                logger.info(f'Added memory nodes for observation based on Storm data')
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
                logger.info(f'Added memory nodes for observation based on Storm data')
        else:
            for obs in range(self.quotient.observations):
                if self.quotient.observation_states[obs]>1:
                    obs_memory_dict[obs] = self.quotient.observation_memory_size[obs] + 1
                else:
                    obs_memory_dict[obs] = 1
            logger.info(f'Increase memory in all imperfect observation')
        self.quotient.set_memory_from_dict(obs_memory_dict)
        family = self.quotient.family

        # if Storm's result is better, use it to obtain main family that considers only the important actions
        if self.storm_control.is_storm_better:
            if self.storm_control.use_cutoffs:
                # consider the cut-off schedulers actions
                result_dict = self.storm_control.result_dict
            else:
                # only consider the induced DTMC actions without cut-off states
                result_dict =self.storm_control.result_dict_no_cutoffs
            main_family = self.storm_control.get_main_restricted_family(family,result_dict)
            subfamily_restrictions = []
            if not self.storm_control.incomplete_exploration:
                subfamily_restrictions = self.storm_control.get_subfamilies_restrictions(family, result_dict)
            subfamilies = self.storm_control.get_subfamilies(subfamily_restrictions, family)
        # if PAYNT is better continue normally
        else:
            main_family = family
            subfamilies = []

        self.synthesizer.subfamilies_buffer = subfamilies
        self.synthesizer.main_family = main_family

        assignment = self.synthesize(family)
        return assignment

    # iterative strategy using Storm analysis to enhance the synthesis

    def strategy_iterative_storm(self, unfold_imperfect_only, unfold_storm=True, qvalues_flag: bool = False):
        '''
        @param unfold_imperfect_only if True, only imperfect observations will be unfolded
        '''
        mem_size = paynt.quotient.pomdp.PomdpQuotient.initial_memory_size
        self.synthesizer.storm_control = self.storm_control
        first = True

        while True:
            assignment = self.unfold_and_synthesize(mem_size,unfold_storm)
            if assignment is not None:
                self.storm_control.latest_paynt_result = assignment
                self.storm_control.paynt_export = self.quotient.extract_policy(
                    assignment)
                self.storm_control.paynt_bounds = self.quotient.specification.optimality.optimum
                self.storm_control.paynt_fsc_size = self.quotient.policy_size(
                    self.storm_control.latest_paynt_result)
                self.storm_control.latest_paynt_result_fsc = self.quotient.assignment_to_fsc(
                    self.storm_control.latest_paynt_result)
                # self.storm_control.qvalues = self.compute_qvalues_for_rl(
                #     assignment=assignment)
            else:
                logging.info("Assignment is None")

                self.storm_control.paynt_fsc_size = self.quotient.policy_size(self.storm_control.latest_paynt_result)
                self.storm_control.latest_paynt_result_fsc = self.quotient.assignment_to_fsc(self.storm_control.latest_paynt_result)
            self.storm_control.update_data()

            if self.synthesis_terminate:
                break

            mem_size += 1

            # break

    def get_agents_wrapper(self) -> AgentsWrapper:
        if not hasattr(self, "agents_wrapper") or self.agents_wrapper is None:
            self.agents_wrapper = AgentsWrapper(
                self.quotient.pomdp, self.args)
        return self.agents_wrapper

    def run_rl_trajectories(self, saynt: bool = True):
        # assignment = self.storm_control.latest_paynt_result
        # qvalues = self.storm_control.qvalues
        fsc = self.storm_control.latest_paynt_result_fsc
        agents_wrapper = self.get_agents_wrapper()
        if saynt:
            original_property = self.quotient.get_property()
            original_property_str = original_property.__str__()
            if RegexPatterns.check_max_property(original_property_str):
                model_reward_multiplier = 1
            else:
                model_reward_multiplier = -1
            if hasattr(self.storm_control, "qvalues"):
                q_values = self.storm_control.qvalues
            else:
                q_values = None
            # q_values = None
            agents_wrapper.get_saynt_trajectories(
                self.storm_control, self.quotient, fsc, q_values, model_reward_multiplier)
            sub_method = self.input_rl_settings_dict["sub_method"]
            agents_wrapper.save_to_json(experiment_name=self.input_rl_settings_dict["agent_task"],
                                        model=self.input_rl_settings_dict["model_name"],
                                        method=f"{self.args.learning_method}_{sub_method}")
            return
        logger.info("Training agent with combination of FSC and RL.")
        agents_wrapper.train_agent_combined_with_fsc_advanced(
            4000, fsc, self.storm_control.paynt_bounds)
        agents_wrapper.save_to_json(experiment_name=self.args.agent_name,
                                    model=self.input_rl_settings_dict["model_name"], method=self.args.learning_method)

    def run_rl_synthesis_critic(self):
        qvalues = self.storm_control.qvalues
        agents_wrapper = AgentsWrapper(
            self.quotient.pomdp, self.args, qvalues=qvalues,
            action_labels_at_observation=self.quotient.action_labels_at_observation)
        agents_wrapper.train_agent(2000)
        agents_wrapper.save_to_json(experiment_name=self.args.agent_name,
                                    model=self.input_rl_settings_dict["model_name"], method=self.args.learning_method)

    def run_rl_synthesis_q_vals_rand(self):
        qvalues = self.storm_control.qvalues
        agents_wrapper = AgentsWrapper(
            self.quotient.pomdp, self.args, qvalues=qvalues,
            action_labels_at_observation=self.quotient.action_labels_at_observation,
            random_init_starts_q_vals=True
        )
        agents_wrapper.train_agent_qval_randomization(2000, qvalues)
        agents_wrapper.save_to_json(experiment_name=self.args.agent_name,
                                    model=self.input_rl_settings_dict["model_name"], method=self.args.learning_method)

    def property_is_reachability(self, property: str):
        return "Pmax" in property or "Pmin" in property

    def property_is_maximizing(self, property: str):
        return "max" in property

    def run_rl_synthesis_behavioral_cloning(self, fsc, save=True, nr_of_iterations=4000):
        sub_method = self.input_rl_settings_dict["sub_method"]
        agents_wrapper = self.get_agents_wrapper()
        agents_wrapper.train_with_bc(
            fsc, sub_method=sub_method, nr_of_iterations=nr_of_iterations)
        experiment_name = f"{self.args.agent_name}_{sub_method}"

        if save:
            self.agents_wrapper.save_to_json(experiment_name=experiment_name,
                                             model=self.input_rl_settings_dict["model_name"],
                                             method=f"{self.args.learning_method}")

    def run_rl_synthesis_jumpstarts(self, fsc, saynt: bool = False, save=True, nr_of_iterations=4000):
        if saynt:
            raise NotImplementedError("SAYNT jumpstarts not implemented yet")
        agents_wrapper = self.get_agents_wrapper()
        agents_wrapper.train_agent_with_jumpstarts(fsc, nr_of_iterations)
        if save:
            agents_wrapper.save_to_json(
                self.args.agent_name, model=self.input_rl_settings_dict["model_name"], method=self.args.learning_method)

    def run_rl_synthesis_shaping(self, fsc, saynt: bool = False, save=True, nr_of_iterations=4000):
        if saynt:
            raise NotImplementedError("SAYNT shaping not implemented yet")
        agents_wrapper = self.get_agents_wrapper()
        agents_wrapper.train_agent_with_shaping(fsc, nr_of_iterations)
        experiment_name = f"{self.args.agent_name}_longer"
        if save:
            agents_wrapper.save_to_json(
                experiment_name, model=self.input_rl_settings_dict["model_name"], method=self.args.learning_method)
        # self.agents_wrapper.save_to_json(experiment_name, model=self.input_rl_settings_dict["model_name"], method=self.args.learning_method)

    # main SAYNT loop
    def iterative_storm_loop_body(self, timeout, paynt_timeout, storm_timeout, iteration_limit=0, rl_family_extraction : bool = False):


    def print_synthesized_controllers(self):
        hline = "\n------------------------------------\n"
        print(hline)
        print("PAYNT results: ")
        print(self.storm_control.paynt_bounds)
        print("controller size: {}".format(self.storm_control.paynt_fsc_size))
        print()
        print("Storm results: ")
        print(self.storm_control.storm_bounds)
        print("controller size: {}".format(self.storm_control.belief_controller_size))
        print(hline)


    def iterative_storm_loop(self, timeout, paynt_timeout, storm_timeout, iteration_limit=0):
        ''' Main SAYNT loop. '''
        self.interactive_queue = Queue()
        self.synthesizer.s_queue = self.interactive_queue
        self.storm_control.interactive_storm_setup()
        iteration = 1
        if rl_family_extraction:
            agents_wrapper = self.get_agents_wrapper()
            agents_wrapper.train_agent(1500)
            if hasattr(self, "input_rl_settings_dict"):
                fsc_size = self.input_rl_settings_dict["fsc_size"]
            else:
                fsc_size = 1
            self.quotient.set_global_memory_size(fsc_size)
            external_family = ExtractedFamilyWrapper(self.quotient.family, fsc_size, agents_wrapper)
        else:
            external_family = None

        paynt_thread = Thread(target=self.strategy_iterative_storm, args=(
            True, self.storm_control.unfold_storm, external_family))

        iteration_timeout = time.time() + timeout

        self.saynt_timer.start()
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

            if iteration == 1:
                self.storm_control.interactive_storm_start(storm_timeout)
            else:
                self.storm_control.interactive_storm_resume(storm_timeout)

            # compute sizes of controllers
            assert self.storm_control.latest_storm_result is not None
            self.storm_control.belief_controller_size = self.storm_control.get_belief_controller_size(self.storm_control.latest_storm_result, self.storm_control.paynt_fsc_size)

            self.print_synthesized_controllers()

            if time.time() > iteration_timeout or iteration == iteration_limit:
                break

            iteration += 1

        self.interactive_queue.put("terminate")
        self.synthesis_terminate = True
        paynt_thread.join()

        self.storm_control.interactive_storm_terminate()

        self.saynt_timer.stop()

    def set_rl_setting(self, input_rl_settings_dict):
        self.saynt = False
        if input_rl_settings_dict["rl_method"] == "BC":
            self.combo_mode = RL_SAYNT_Combo_Modes.BEHAVIORAL_CLONING
        elif input_rl_settings_dict["rl_method"] == "Trajectories":
            self.combo_mode = RL_SAYNT_Combo_Modes.TRAJECTORY_MODE
        elif input_rl_settings_dict["rl_method"] == "SAYNT_Trajectories":
            self.combo_mode = RL_SAYNT_Combo_Modes.TRAJECTORY_MODE
            self.saynt = True
        elif input_rl_settings_dict["rl_method"] == "JumpStarts":
            self.combo_mode = RL_SAYNT_Combo_Modes.JUMPSTART_MODE
        elif input_rl_settings_dict["rl_method"] == "R_Shaping":
            self.combo_mode = RL_SAYNT_Combo_Modes.SHAPING_MODE
        else:
            self.combo_mode = RL_SAYNT_Combo_Modes.BEHAVIORAL_CLONING

    def iterative_storm_loop(self, timeout, paynt_timeout, storm_timeout, iteration_limit=0):
        self.run_rl = False
        if hasattr(self, "input_rl_settings_dict"):
            if self.input_rl_settings_dict["reinforcement_learning"]:
                self.run_rl = True
                self.set_rl_setting(self.input_rl_settings_dict)
                self.args = init_rl_args(mode=self.combo_mode)
        

        self.try_faster = False
        
        skip = False
        if self.try_faster and hasattr(self, "input_rl_settings_dict"):
            agent_task = self.input_rl_settings_dict["agent_task"]
            model_name = self.input_rl_settings_dict["model_name"]
            fsc_file_name = f"{model_name}"
            if os.path.exists(f"{agent_task}/{fsc_file_name}") and os.path.exists(f"{agent_task}/{fsc_file_name}_info.json"):
                with open(f"{agent_task}/{fsc_file_name}", "rb") as f:
                    fsc = pickle.load(f)
                with open(f"{agent_task}/{fsc_file_name}_info.json", "r") as f:
                    fsc_json_dict = json.load(f)
                self.storm_control.latest_paynt_result_fsc = fsc
                skip = True
        if not skip:

            self.iterative_storm_loop_body(
                timeout, paynt_timeout, storm_timeout, iteration_limit, rl_family_extraction=False)
            specification_property = self.quotient.get_property().__str__()
            paynt_value = self.storm_control.paynt_bounds
            storm_value = self.storm_control.storm_bounds
            fsc_json_dict = {"specification_property": specification_property,
                             "paynt_value": paynt_value, "storm_value": storm_value}
            if self.try_faster and hasattr(self, "input_rl_settings_dict"):

                with open(f"{agent_task}/{fsc_file_name}", "wb") as f:
                    pickle.dump(self.storm_control.latest_paynt_result_fsc, f)
                with open(f"{agent_task}/{fsc_file_name}_info.json", "w") as f:
                    json.dump(fsc_json_dict, f)

        if self.run_rl and self.combo_mode == RL_SAYNT_Combo_Modes.TRAJECTORY_MODE:
            self.run_rl_trajectories(self.saynt)
        elif self.run_rl and self.combo_mode == RL_SAYNT_Combo_Modes.QVALUES_CRITIC_MODE:
            self.run_rl_synthesis_critic()
        elif self.run_rl and self.combo_mode == RL_SAYNT_Combo_Modes.QVALUES_RANDOM_SIM_INIT_MODE:
            self.run_rl_synthesis_q_vals_rand()
        elif self.run_rl and self.combo_mode == RL_SAYNT_Combo_Modes.BEHAVIORAL_CLONING:
            self.run_rl_synthesis_behavioral_cloning(fsc_json_dict)
        elif self.run_rl and self.combo_mode == RL_SAYNT_Combo_Modes.JUMPSTART_MODE:
            self.run_rl_synthesis_jumpstarts(
                self.storm_control.latest_paynt_result_fsc, False)
        elif self.run_rl and self.combo_mode == RL_SAYNT_Combo_Modes.SHAPING_MODE:
            self.run_rl_synthesis_shaping(
                self.storm_control.latest_paynt_result_fsc, False)

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

    def set_memory_original(self, mem_size):
        """ Set memory size based on the original PAYNT implementation.

        Args:
            mem_size (int): Memory size for condition.
        """
        if mem_size > 1:
            obs_memory_dict = {}
            if self.storm_control.is_storm_better:
                if self.storm_control.is_memory_needed():
                    obs_memory_dict = self.storm_control.memory_vector
                    logger.info(
                        f'Added memory nodes for observation based on Storm data')
                else:
                    # consider the cut-off schedulers actions when updating memory
                    if self.storm_control.unfold_cutoff:
                        for obs in range(self.quotient.observations):
                            if obs in self.storm_control.result_dict:
                                obs_memory_dict[obs] = self.quotient.observation_memory_size[obs] + 1
                            else:
                                obs_memory_dict[obs] = self.quotient.observation_memory_size[obs]
                    # only consider the induced DTMC without cut-off states
                    else:
                        for obs in range(self.quotient.observations):
                            if obs in self.storm_control.result_dict_no_cutoffs:
                                obs_memory_dict[obs] = self.quotient.observation_memory_size[obs] + 1
                            else:
                                obs_memory_dict[obs] = self.quotient.observation_memory_size[obs]
                    logger.info(
                        f'Added memory nodes for observation based on Storm data')
            else:
                for obs in range(self.quotient.observations):
                    if self.quotient.observation_states[obs] > 1:
                        obs_memory_dict[obs] = self.quotient.observation_memory_size[obs] + 1
                    else:
                        obs_memory_dict[obs] = 1
                logger.info(f'Increase memory in all imperfect observation')
            self.quotient.set_memory_from_dict(obs_memory_dict)

    def set_advices_from_rl(self, interpretation_result=None, family=None):
        """ Set advices from RL interpretation.

        Args:
            interpretation_result (tuple, optional): Tuple of dictionaries (obs_act_dict, memory_dict, labels). Defaults to None.
            load_rl_dict (bool, optional): Whether to load RL interpretation from a file. Defaults to True.
            rl_dict (bool, optional): Whether to use RL interpretation. Defaults to False.
            family (paynt.quotient.quotient.QuotientFamily, optional): Family of the POMDP. Defaults to None.
        """
        obs_actions = interpretation_result[0]
        self.memory_dict = interpretation_result[1]
        action_keywords = interpretation_result[2]
        self.priority_list = interpretation_result[3]
        obs_actions = self.storm_control.convert_rl_dict_to_paynt(
            family, obs_actions, action_keywords)
        self.storm_control.result_dict = obs_actions
        self.storm_control.result_dict_no_cutoffs = obs_actions

    def set_reinforcement_learning(self, input_rl_settings_dict):
        """ Set RL settings from PAYNT cli.
        Args:
            input_rl_settings_dict (dict): Dictionary with RL settings from PAYNT cli.
        """
        self.input_rl_settings_dict = input_rl_settings_dict

    def set_input_rl_settings_for_paynt(self, input_rl_settings_dict):
        self.set_rl_setting(self.input_rl_settings_dict)
        self.use_rl = True
        self.loop = self.input_rl_settings_dict["loop"]
        self.rl_training_iters = self.input_rl_settings_dict['rl_training_iters']
        self.rl_load_memory_flag = self.input_rl_settings_dict['rl_load_memory_flag']
        self.greedy = self.input_rl_settings_dict['greedy']
        if self.loop:
            self.fsc_synthesis_time_limit = self.input_rl_settings_dict['fsc_time_in_loop']
        else:
            self.fsc_synthesis_time_limit = self.input_rl_settings_dict['time_limit']
        # if self.loop:
        self.time_limit = self.input_rl_settings_dict['time_limit']
        self.rnn_less = self.input_rl_settings_dict['rnn_less']

    # PAYNT POMDP synthesis that uses pre-computed results from Storm as guide

    def get_agent_name(self):
        agent_name = "PAYNT_w_Advices"
        if self.loop:
            agent_name += "_loop_" + self.input_rl_settings_dict["rl_method"]
        if self.rl_load_memory_flag:
            agent_name += "_w_Memory"
        if self.greedy:
            agent_name += "_greedy"
        return agent_name

    def genearate_args(self, agent_name, rnn_less = False, nr_runs = 2001) -> ArgsEmulator:
        # nr_runs = self.rl_training_iters
        # nr_runs = 2001
        return ArgsEmulator(learning_rate=1.6e-4,
                            restart_weights=0, learning_method="Stochastic_PPO",
                            nr_runs=nr_runs, agent_name=agent_name, load_agent=False,
                            evaluate_random_policy=False, max_steps=400, evaluation_goal=50, evaluation_antigoal=-20,
                            trajectory_num_steps=32, discount_factor=0.99, num_environments=256,
                            normalize_simulator_rewards=False, buffer_size=500, random_start_simulator=True,
                            batch_size=256, vectorized_envs_flag=True, perform_interpretation=True, use_rnn_less=rnn_less, model_memory_size=0)

    def perform_rl_training_w_fsc(self, fsc):
        if self.combo_mode == RL_SAYNT_Combo_Modes.JUMPSTART_MODE:
            logger.info("Training agent with jumpstarts.")
            self.run_rl_synthesis_jumpstarts(
                fsc, False, save=False, nr_of_iterations=self.args.nr_runs)
        elif self.combo_mode == RL_SAYNT_Combo_Modes.SHAPING_MODE:
            logger.info("Training agent with shaping.")
            self.run_rl_synthesis_shaping(
                fsc, False, save=False, nr_of_iterations=self.args.nr_runs)
        elif self.combo_mode == RL_SAYNT_Combo_Modes.BEHAVIORAL_CLONING:
            logger.info("Training agent with behavioral cloning.")
            self.run_rl_synthesis_behavioral_cloning(
                fsc, save=False, nr_of_iterations=self.args.nr_runs)
        else:
            self.agents_wrapper.train_agent(self.args.nr_runs)

    def train_and_interpret(self, fsc):
        if fsc is not None:
            self.perform_rl_training_w_fsc(fsc)
        else:
            logger.info("FSC is None. Training agent without FSC.")
            self.agents_wrapper.train_agent(self.args.nr_runs)
        logger.info("Interpreting agent ...")
        interpretation_result = self.agents_wrapper.interpret_agent(
                best=False, randomize_illegal_actions=True, greedy=self.greedy)
        logger.info("Interpretation finished.")
        return interpretation_result
    
    def set_memory_rl_loop(self):
        logger.info("Adding memory nodes based on RL interpretation.")
        priority_list_len = int(
                    math.ceil(len(self.priority_list) / 5))
        priorities = self.priority_list[:priority_list_len]
        for i in range(len(priorities)):
            self.memory_dict[priorities[i]] += 1
        self.quotient.set_memory_from_dict(self.memory_dict)
    
    def set_memory_storm(self, mem_size, unfold_storm=True, interpretation_result=None, odd=False, unfold_imperfect_only=False):
        # unfold memory according to the best result
            if (not self.use_rl or interpretation_result is None) and unfold_storm:
                self.set_memory_original(mem_size)
            elif self.use_rl and self.rl_load_memory_flag and odd:
                self.set_memory_rl_loop()
            elif self.use_rl and self.rl_load_memory_flag and not odd:
                logger.info("Adding memory nodes based to each observation.")
                for i in self.memory_dict:
                    self.memory_dict[i] += 1
            else:
                logger.info(
                    "Synthesizing optimal k={} controller ...".format(mem_size))
                if unfold_imperfect_only:
                    self.quotient.set_imperfect_memory_size(mem_size)
                else:
                    self.quotient.set_global_memory_size(mem_size)

    def initialize_rl_storm(self, use_rl):
        if use_rl:
            agent_name = self.get_agent_name()
            self.args = self.genearate_args(agent_name, self.rnn_less, self.rl_training_iters)
            self.agents_wrapper = AgentsWrapper(
                self.quotient.pomdp, self.args)
    
    def train_agent(self, nr_runs):
        self.agents_wrapper.train_agent(nr_runs)
        interpretation_result = self.agents_wrapper.interpret_agent(
            best=False, greedy=self.greedy)
        return interpretation_result
    
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
            assignment = self.unfold_and_synthesize(mem_size,unfold_storm)
            if assignment is not None:
                self.storm_control.latest_paynt_result = assignment
                self.storm_control.paynt_export = self.quotient.extract_policy(
                    assignment)
                self.storm_control.paynt_bounds = self.quotient.specification.optimality.optimum
                if hasattr(self, "agents_wrapper"):
                    self.agents_wrapper.agent.evaluation_result.add_paynt_bound(
                        self.storm_control.paynt_bounds)

            self.storm_control.update_data()
            mem_size += 1

        if hasattr(self, "agents_wrapper"):
            self.agents_wrapper.save_to_json(
                self.args.agent_name, model=self.input_rl_settings_dict["model_name"], method=self.args.learning_method, time=end_time - start_time)

    def strategy_iterative(self, unfold_imperfect_only):
        '''
        @param unfold_imperfect_only if True, only imperfect observations will be unfolded
        '''
        mem_size = paynt.quotient.pomdp.PomdpQuotient.initial_memory_size
        opt = self.quotient.specification.optimality.optimum
        assignment_last = None
        while True:

            logger.info("Synthesizing optimal k={} controller ...".format(mem_size) )
            if unfold_imperfect_only:
                self.quotient.set_imperfect_memory_size(mem_size)
            else:
                self.quotient.set_global_memory_size(mem_size)

            self.synthesize(self.quotient.family)

            self.synthesize(self.quotient.family)
            assignment = self.synthesize(self.quotient.design_space)
            if assignment is not None:
                assignment_last = assignment
            opt_old = opt
            opt = self.quotient.specification.optimality.optimum

            # finish if optimum has not been improved
            # if opt_old == opt and opt is not None:
            #     break
            mem_size += 1
            #break

    def run(self, optimum_threshold=None):
        if hasattr(self, "input_rl_settings_dict"):
            synthesizer_rl = SynthesizerRL(self.quotient, self.method, self.storm_control, self.input_rl_settings_dict)
            result = synthesizer_rl.run()
            return result
        if self.storm_control is None:
            # Pure PAYNT POMDP synthesis
            self.strategy_iterative(unfold_imperfect_only=True)
            return

        # SAYNT
        logger.info("Storm POMDP option enabled")
        logger.info("Storm settings: iterative - {}, get_storm_result - {}, storm_options - {}, prune_storm - {}, unfold_strategy - {}, use_storm_cutoffs - {}".format(
                    (self.storm_control.iteration_timeout, self.storm_control.paynt_timeout, self.storm_control.storm_timeout), self.storm_control.get_result,
                    self.storm_control.storm_options, self.storm_control.incomplete_exploration, (self.storm_control.unfold_storm, self.storm_control.unfold_cutoff), self.storm_control.use_cutoffs
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
            self.strategy_storm(unfold_imperfect_only=True, unfold_storm=self.storm_control.unfold_storm)

        self.print_synthesized_controllers()
