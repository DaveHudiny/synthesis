import json
import numpy as np
from time import sleep
import os
from rl_src.experimental_interface import ArgsEmulator
import pickle
import stormpy

from .statistic import Statistic
import paynt.synthesizer.synthesizer_ar
from .synthesizer_ar_storm import SynthesizerARStorm
from .synthesizer_hybrid import SynthesizerHybrid
from .synthesizer_multicore_ar import SynthesizerMultiCoreAR
from .synthesizer_rl import Synthesizer_RL

from paynt.rl_extension.saynt_rl_tools.q_values_correction import make_qvalues_table_tensorable
from ..rl_extension.saynt_rl_tools.rl_saynt_combo_modes import RL_SAYNT_Combo_Modes, init_rl_args

from paynt.rl_extension.saynt_rl_tools.regex_patterns import RegexPatterns
from paynt.parser.prism_parser import PrismParser

import paynt.quotient.quotient
import paynt.quotient.pomdp
import paynt.utils.timer

import paynt.verification.property

import math
from collections import defaultdict

from threading import Thread
from queue import Queue
import time

import logging
logger = logging.getLogger(__name__)

from rl_src.interpreters.fsc_based_interpreter import ExtractedFSCPolicy
from rl_src.tools.evaluators import evaluate_extracted_fsc


class SynthesizerPomdp:

    # If true explore only the main family
    incomplete_exploration = False

    def __init__(self, quotient, method, storm_control):
        self.quotient = quotient
        self.use_storm = False
        self.synthesizer = None
        if method == "ar":
            self.synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR
        elif method == "ar_multicore":
            self.synthesizer = SynthesizerMultiCoreAR
        elif method == "hybrid":
            self.synthesizer = SynthesizerHybrid
        self.total_iters = 0

        if storm_control is not None:
            self.use_storm = True
            self.storm_control = storm_control
            self.storm_control.quotient = self.quotient
            self.storm_control.pomdp = self.quotient.pomdp
            self.storm_control.spec_formulas = self.quotient.specification.stormpy_formulae()
            self.synthesis_terminate = False
            # SAYNT only works with abstraction refinement
            self.synthesizer = SynthesizerARStorm
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

    # iterative strategy using Storm analysis to enhance the synthesis

    def strategy_iterative_storm(self, unfold_imperfect_only, unfold_storm=True, qvalues_flag: bool = False):
        '''
        @param unfold_imperfect_only if True, only imperfect observations will be unfolded
        '''
        mem_size = paynt.quotient.pomdp.PomdpQuotient.initial_memory_size

        self.synthesizer.storm_control = self.storm_control

        while True:
            # for x in range(2):

            paynt.quotient.pomdp.PomdpQuotient.current_family_index = mem_size

            # unfold memory according to the best result
            if unfold_storm:
                if mem_size > 1:
                    obs_memory_dict = {}
                    if self.storm_control.is_storm_better:
                        # Storm's result is better and it needs memory
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
                        logger.info(
                            f'Increase memory in all imperfect observation')
                    self.quotient.set_memory_from_dict(obs_memory_dict)
            else:
                logger.info(
                    "Synthesizing optimal k={} controller ...".format(mem_size))
                if unfold_imperfect_only:
                    self.quotient.set_imperfect_memory_size(mem_size)
                else:
                    self.quotient.set_global_memory_size(mem_size)

            family = self.quotient.family

            # if Storm's result is better, use it to obtain main family that considers only the important actions
            if self.storm_control.is_storm_better:
                # consider the cut-off schedulers actions
                if self.storm_control.use_cutoffs:
                    main_family = self.storm_control.get_main_restricted_family(
                        family, self.storm_control.result_dict)
                    if self.storm_control.incomplete_exploration == True:
                        subfamily_restrictions = []
                    else:
                        subfamily_restrictions = self.storm_control.get_subfamilies_restrictions(
                            family, self.storm_control.result_dict)
                # only consider the induced DTMC actions without cut-off states
                else:
                    main_family = self.storm_control.get_main_restricted_family(
                        family, self.storm_control.result_dict_no_cutoffs)
                    if self.storm_control.incomplete_exploration == True:
                        subfamily_restrictions = []
                    else:
                        subfamily_restrictions = self.storm_control.get_subfamilies_restrictions(
                            family, self.storm_control.result_dict_no_cutoffs)

                subfamilies = self.storm_control.get_subfamilies(
                    subfamily_restrictions, family)
            # if PAYNT is better continue normally
            else:
                main_family = family
                subfamilies = []

            self.synthesizer.subfamilies_buffer = subfamilies
            self.synthesizer.main_family = main_family

            assignment = self.synthesize(family)

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

            self.storm_control.update_data()

            if self.synthesis_terminate:
                break

            mem_size += 1

            # break

    def run_rl_trajectories(self, saynt: bool = True):
        # assignment = self.storm_control.latest_paynt_result
        # qvalues = self.storm_control.qvalues
        fsc = self.storm_control.latest_paynt_result_fsc
        rl_synthesiser = Synthesizer_RL(self.quotient.pomdp, self.args)
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
            rl_synthesiser.get_saynt_trajectories(
                self.storm_control, self.quotient, fsc, q_values, model_reward_multiplier)
            sub_method = self.input_rl_settings_dict["sub_method"]
            rl_synthesiser.save_to_json(experiment_name=self.input_rl_settings_dict["agent_task"],
                                        model=self.input_rl_settings_dict["model_name"],
                                        method=f"{self.args.learning_method}_{sub_method}")
            return
        logger.info("Training agent with combination of FSC and RL.")
        rl_synthesiser.train_agent_combined_with_fsc_advanced(
            4000, fsc, self.storm_control.paynt_bounds)
        rl_synthesiser.save_to_json(experiment_name=self.args.agent_name,
                                    model=self.input_rl_settings_dict["model_name"], method=self.args.learning_method)

    def run_rl_synthesis_critic(self):
        qvalues = self.storm_control.qvalues
        rl_synthesiser = Synthesizer_RL(
            self.quotient.pomdp, self.args, qvalues=qvalues,
            action_labels_at_observation=self.quotient.action_labels_at_observation)
        rl_synthesiser.train_agent(2000)
        rl_synthesiser.save_to_json(experiment_name=self.args.agent_name,
                                    model=self.input_rl_settings_dict["model_name"], method=self.args.learning_method)

    def run_rl_synthesis_q_vals_rand(self):
        qvalues = self.storm_control.qvalues
        rl_synthesizer = Synthesizer_RL(
            self.quotient.pomdp, self.args, qvalues=qvalues,
            action_labels_at_observation=self.quotient.action_labels_at_observation,
            random_init_starts_q_vals=True
        )
        rl_synthesizer.train_agent_qval_randomization(2000, qvalues)
        rl_synthesizer.save_to_json(experiment_name=self.args.agent_name,
                                    model=self.input_rl_settings_dict["model_name"], method=self.args.learning_method)

    def property_is_reachability(self, property: str):
        return "Pmax" in property or "Pmin" in property

    def property_is_maximizing(self, property: str):
        return "max" in property

    def run_rl_synthesis_behavioral_cloning(self, fsc, save=True, nr_of_iterations=4000):
        sub_method = self.input_rl_settings_dict["sub_method"]
        if not hasattr(self, "rl_synthesiser"):
            self.rl_synthesiser = Synthesizer_RL(
                self.quotient.pomdp, self.args, pretrain_dqn=True)
        self.rl_synthesiser.train_with_bc(
            fsc, sub_method=sub_method, nr_of_iterations=nr_of_iterations)
        experiment_name = f"{self.args.agent_name}_{sub_method}"

        if save:
            self.rl_synthesiser.save_to_json(experiment_name=experiment_name,
                                             model=self.input_rl_settings_dict["model_name"],
                                             method=f"{self.args.learning_method}")

    def run_rl_synthesis_jumpstarts(self, fsc, saynt: bool = False, save=True, nr_of_iterations=4000):
        if saynt:
            raise NotImplementedError("SAYNT jumpstarts not implemented yet")
        if not hasattr(self, "rl_synthesiser"):
            self.rl_synthesiser = Synthesizer_RL(
                self.quotient.pomdp, self.args)
        self.rl_synthesiser.train_agent_with_jumpstarts(fsc, nr_of_iterations)
        if save:
            self.rl_synthesiser.save_to_json(
                self.args.agent_name, model=self.input_rl_settings_dict["model_name"], method=self.args.learning_method)

    def run_rl_synthesis_shaping(self, fsc, saynt: bool = False, save=True, nr_of_iterations=4000):
        if saynt:
            raise NotImplementedError("SAYNT shaping not implemented yet")
        if not hasattr(self, "rl_synthesiser"):

            self.rl_synthesiser = Synthesizer_RL(
                self.quotient.pomdp, self.args)
        self.rl_synthesiser.train_agent_with_shaping(fsc, nr_of_iterations)
        experiment_name = f"{self.args.agent_name}_longer"
        if save:
            self.rl_synthesiser.save_to_json(
                experiment_name, model=self.input_rl_settings_dict["model_name"], method=self.args.learning_method)
        # self.rl_synthesiser.save_to_json(experiment_name, model=self.input_rl_settings_dict["model_name"], method=self.args.learning_method)

    # main SAYNT loop
    def iterative_storm_loop_body(self, timeout, paynt_timeout, storm_timeout, iteration_limit=0):
        self.interactive_queue = Queue()
        self.synthesizer.s_queue = self.interactive_queue
        self.storm_control.interactive_storm_setup()
        iteration = 1
        paynt_thread = Thread(target=self.strategy_iterative_storm, args=(
            True, self.storm_control.unfold_storm))

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
            self.storm_control.belief_controller_size = self.storm_control.get_belief_controller_size(
                self.storm_control.latest_storm_result, self.storm_control.paynt_fsc_size)

            print("\n------------------------------------\n")
            print("PAYNT results: ")
            print(self.storm_control.paynt_bounds)
            print("controller size: {}".format(
                self.storm_control.paynt_fsc_size))

            print()

            print("Storm results: ")
            print(self.storm_control.storm_bounds)
            print("controller size: {}".format(
                self.storm_control.belief_controller_size))
            print("\n------------------------------------\n")

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
                timeout, paynt_timeout, storm_timeout, iteration_limit)
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
        self.fsc_synthesis_time_limit = self.input_rl_settings_dict['fsc_time_in_loop']
        self.time_limit = self.input_rl_settings_dict['time_limit']

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

    def genearate_args(self, agent_name) -> ArgsEmulator:
        # nr_runs = self.rl_training_iters
        nr_runs = 201
        return ArgsEmulator(learning_rate=1.6e-4,
                            restart_weights=0, learning_method="Stochastic_PPO",
                            nr_runs=nr_runs, agent_name=agent_name, load_agent=False,
                            evaluate_random_policy=False, max_steps=400, evaluation_goal=50, evaluation_antigoal=-20,
                            trajectory_num_steps=32, discount_factor=0.99, num_environments=256,
                            normalize_simulator_rewards=False, buffer_size=500, random_start_simulator=False,
                            batch_size=256, vectorized_envs_flag=True, perform_interpretation=True, use_rnn_less=True, model_memory_size=0)

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
            self.rl_synthesiser.train_agent(self.args.nr_runs)

    def train_and_interpret(self, fsc):
        if fsc is not None:
            self.perform_rl_training_w_fsc(fsc)
        else:
            logger.info("FSC is None. Training agent without FSC.")
            self.rl_synthesiser.train_agent(self.args.nr_runs)
        logger.info("Interpreting agent ...")
        interpretation_result = self.rl_synthesiser.interpret_agent(
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

    def get_vector_and_memory_from_family(self, family, hole):
        import re
        name = family.hole_name(hole)
        if name[0] == "M":
            is_update = True
        else:
            is_update = False
        vector_match = re.search(r'\[([^\]]+)\]', name)
        if vector_match:
            vector_string = vector_match.group(1)
            vector = []
            for item in vector_string.split('&'):
                if "=" not in item:
                    if "!" in item:
                        vector.append(float(-1.0))
                    else:
                        vector.append(float(1.0)) 
                else:
                    _, value = item.split('=')
                    value = value.strip()
                    vector.append(float(value)) 
        last_number_match = re.search(r',(\d+)\)$', name)
        if last_number_match:
            last_number = int(last_number_match.group(1))
        return vector, last_number, is_update
    
    def get_restricted_family_rl_inference(self, original_family):
        restricted_family = original_family.copy()
        subfamily_restrictions = []

        # Statistics 
        num_misses = 0
        num_misses_complete = 0

        logger.info("Building family from FFNN...")
        
        # Extraction and evaluation of original policy
        extracted_fsc_policy = ExtractedFSCPolicy(self.rl_synthesiser.agent.wrapper, self.rl_synthesiser.agent.environment,
                                                  tf_environment=self.rl_synthesiser.agent.tf_environment, args = self.args)
        evaluate_extracted_fsc(external_evaluation_result=self.rl_synthesiser.agent.evaluation_result, 
                               agent=self.rl_synthesiser.agent,
                               extracted_fsc_policy=extracted_fsc_policy)
        
        for hole in range(restricted_family.num_holes):
            vector, mem_number, is_mem_update = self.get_vector_and_memory_from_family(restricted_family, hole)
            
            observation_integer = restricted_family.get_hole_observation_index(hole)
            options = restricted_family.hole_options(hole)
            labels = [str(restricted_family.hole_to_option_labels[hole][option]) for option in options]
            if is_mem_update:
                restricted_family.hole_set_options(hole, [0])
                restriction = {"hole": hole, "restriction": [0]}
                subfamily_restrictions.append(restriction)
                continue
            elif mem_number > 0:
                # subfamily_restrictions.append({"hole": hole, "restriction": []})
                continue
            fake_time_step = self.rl_synthesiser.agent.environment.create_fake_timestep_from_observation_integer(observation_integer)
            i = 0
            while True:
                if i == 0:
                    action = extracted_fsc_policy.get_single_action(observation_integer, mem_number)
                    action_label = self.rl_synthesiser.agent.environment.act_to_keywords[action]
                else:
                    action = self.rl_synthesiser.agent.wrapper.action(fake_time_step)
                    extracted_fsc_policy.set_single_action(observation_integer, mem_number, action.action.numpy()[0])
                    action_label = self.rl_synthesiser.agent.environment.act_to_keywords[action.action.numpy()[0]]
                if action_label in labels:
                    index_of_action_label = labels.index(action_label)
                    restricted_family.hole_set_options(hole, [index_of_action_label])
                    restriction = {"hole": hole, "restriction": [index_of_action_label]}
                    subfamily_restrictions.append(restriction)
                    break
                elif i == 0:
                    num_misses += 1
                elif i > 10:
                # else:
                    selected_action = np.random.choice(options)
                    extracted_fsc_policy.set_single_action(observation_integer, mem_number, selected_action)
                    restricted_family.hole_set_options(hole, [selected_action])
                    restriction = {"hole": hole, "restriction": [selected_action]}
                    subfamily_restrictions.append(restriction)
                    num_misses_complete += 1
                    break
                i += 1
        extracted_fsc_policy.recompile_tf_tables()
        evaluate_extracted_fsc(external_evaluation_result=self.rl_synthesiser.agent.evaluation_result,
                               agent=self.rl_synthesiser.agent, extracted_fsc_policy=extracted_fsc_policy)
        logger.info(f"Number of misses: {num_misses} out of {restricted_family.num_holes}")
        logger.info(f"Number of complete misses: {num_misses_complete} out of {restricted_family.num_holes}")
        return restricted_family, subfamily_restrictions
    
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
            self.args = self.genearate_args(agent_name)
            self.rl_synthesiser = Synthesizer_RL(
                self.quotient.pomdp, self.args)
    
    def train_agent(self, nr_runs):
        self.rl_synthesiser.train_agent(nr_runs)
        interpretation_result = self.rl_synthesiser.interpret_agent(
            best=False, greedy=self.greedy)
        return interpretation_result
    
    def strategy_storm(self, unfold_imperfect_only, unfold_storm=True):
        '''
        @param unfold_imperfect_only if True, only imperfect observations will be unfolded
        '''
        # mem_size = paynt.quotient.pomdp.PomdpQuotient.initial_memory_size
        paynt.quotient.pomdp.PomdpQuotient.initial_memory_size = 3
        mem_size = paynt.quotient.pomdp.PomdpQuotient.initial_memory_size
        self.synthesizer.storm_control = self.storm_control
        self.use_rl, self.loop = False, False
        if hasattr(self, 'input_rl_settings_dict'):
            self.set_input_rl_settings_for_paynt(self.input_rl_settings_dict)
        first_run = True
        current_time = None
        if self.loop:
            current_time = self.fsc_synthesis_time_limit
        interpretation_result = None
        self.rl_synthesiser = None
        start_time = time.time()
        self.initialize_rl_storm(self.use_rl)
        if not self.loop and self.use_rl:
            # pass
            interpretation_result = self.train_agent(self.args.nr_runs)

        fsc = None
        odd = False
        while True:
            self.quotient.set_global_memory_size(mem_size)
            if self.use_rl:
                current_time = self.fsc_synthesis_time_limit
            if self.use_rl and self.loop and not first_run:
                interpretation_result = self.train_and_interpret(fsc)
            if self.storm_control.is_storm_better == False:
                self.storm_control.parse_results(self.quotient)

            paynt.quotient.pomdp.PomdpQuotient.current_family_index = mem_size
            if interpretation_result is not None:
                self.memory_dict = interpretation_result[1]
                self.priority_list = interpretation_result[3]

            self.set_memory_storm(mem_size, unfold_storm, interpretation_result, odd, unfold_imperfect_only)


            family = self.quotient.family
            
            # TODO: Add condition, as this option removes the ability to work with the original family properly and replaces it with a restricted one
            if self.use_rl:
                pseudo_family, pseudo_subfamily_restrictions = self.get_restricted_family_rl_inference(family)


            if self.use_rl and interpretation_result is not None:
                self.set_advices_from_rl(interpretation_result=interpretation_result,
                                         family=family)

            # if Storm's result is better, use it to obtain main family that considers only the important actions
            if self.storm_control.is_storm_better:
                # consider the cut-off schedulers actions
                if self.storm_control.use_cutoffs:
                    main_family = self.storm_control.get_main_restricted_family(
                        family, self.storm_control.result_dict)
                    if self.storm_control.incomplete_exploration == True:
                        subfamily_restrictions = []
                    else:
                        subfamily_restrictions = self.storm_control.get_subfamilies_restrictions(
                            family, self.storm_control.result_dict)
                # only consider the induced DTMC actions without cut-off states
                else:
                    main_family = self.storm_control.get_main_restricted_family(
                        family, self.storm_control.result_dict_no_cutoffs)
                    if self.storm_control.incomplete_exploration == True:
                        subfamily_restrictions = []
                    else:
                        subfamily_restrictions = self.storm_control.get_subfamilies_restrictions(
                            family, self.storm_control.result_dict_no_cutoffs)
                if self.use_rl:
                    subfamily_restrictions = pseudo_subfamily_restrictions
                subfamilies = self.storm_control.get_subfamilies(
                    subfamily_restrictions, family)
            # if PAYNT is better continue normally
            else:
                main_family = family
                subfamilies = []
            self.synthesizer.subfamilies_buffer = subfamilies

            if self.use_rl:
                self.synthesizer.main_family = pseudo_family # main_family

            if self.loop:
                print("Synthesizing optimal k={} controller ...".format(mem_size))
                print("Time limit: {}".format(current_time))
                assignment = self.synthesize(family, timer=360)
                self.quotient.build()
                try:
                    new_fsc = self.quotient.assignment_to_fsc(assignment)
                    if new_fsc is not None:
                        fsc = new_fsc
                except:
                    logger.info(
                        "FSC could not be created from the assignment. Probably no improvement.")
            else:
                assignment = self.synthesize(family, timer=1800)

            if assignment is not None:
                self.storm_control.latest_paynt_result = assignment
                self.storm_control.paynt_export = self.quotient.extract_policy(
                    assignment)
                self.storm_control.paynt_bounds = self.quotient.specification.optimality.optimum
                if hasattr(self, "rl_synthesiser"):
                    self.rl_synthesiser.agent.evaluation_result.add_paynt_bound(
                        self.storm_control.paynt_bounds)

            first_run = False
            self.storm_control.update_data()

            # TODO: Add condition, as this option removes the ability to increase the memory during FSC synthesis.
            break
            mem_size += 1
            odd = not odd
            if mem_size >= 8 or time.time() - start_time > self.time_limit:
                break
            
        end_time = time.time()
        logger.info(f"Total time: {end_time - start_time}")

        if hasattr(self, "rl_synthesiser"):
            self.rl_synthesiser.save_to_json(
                self.args.agent_name, model=self.input_rl_settings_dict["model_name"], method=self.args.learning_method, time=end_time - start_time)

    def strategy_iterative(self, unfold_imperfect_only):
        '''
        @param unfold_imperfect_only if True, only imperfect observations will be unfolded
        '''
        mem_size = paynt.quotient.pomdp.PomdpQuotient.initial_memory_size
        opt = self.quotient.specification.optimality.optimum
        start_time = time.time()
        time_limit = 10
        assignment_last = None
        while True:
            if time.time() - start_time > time_limit:
                return assignment_last
            logger.info(
                "Synthesizing optimal k={} controller ...".format(mem_size))
            if unfold_imperfect_only:
                self.quotient.set_imperfect_memory_size(mem_size)
            else:
                self.quotient.set_global_memory_size(mem_size)

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

            # break

    def run(self, optimum_threshold=None):
        # choose the synthesis strategy:
        if self.use_storm:
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
                return self.strategy_storm(unfold_imperfect_only=True, unfold_storm=self.storm_control.unfold_storm)

            print("\n------------------------------------\n")
            print("PAYNT results: ")
            print(self.storm_control.paynt_bounds)
            print("controller size: {}".format(
                self.storm_control.paynt_fsc_size))

            print()

            print("Storm results: ")
            print(self.storm_control.storm_bounds)
            print("controller size: {}".format(
                self.storm_control.belief_controller_size))
            print("\n------------------------------------\n")
        # Pure PAYNT POMDP synthesis
        else:
            # self.strategy_iterative(unfold_imperfect_only=False)
            return self.strategy_iterative(unfold_imperfect_only=True)
