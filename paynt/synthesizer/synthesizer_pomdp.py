import json
import numpy as np
from time import sleep
import os
from rl_src.rl_main import ArgsEmulator
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

    def synthesize(self, family=None, print_stats=True):
        if family is None:
            family = self.quotient.family
        synthesizer = self.synthesizer(self.quotient)
        family.constraint_indices = self.quotient.family.constraint_indices
        assignment = synthesizer.synthesize(
            family, keep_optimum=True, print_stats=print_stats)
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
                original_qvalues, self.rl_args.evaluation_goal)
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
                f"R{{\"{reward_model_name}\"}}min=? [ C<={self.rl_args.max_steps} ]", prism, assignment)
            qvalues = qvalues + trap_qvalues * \
                self.rl_args.evaluation_antigoal - cum_reward_qvalues
        elif "Pmin" in original_property_str:
            # TODO: Make proper Pmin correction
            qvalues = self.rl_args.evaluation_antigoal * original_qvalues
        elif RegexPatterns.check_max_property(original_property_str):
            qvalues = original_qvalues + self.rl_args.evaluation_goal
        elif RegexPatterns.check_min_property(original_property_str):
            qvalues = (-original_qvalues) + self.rl_args.evaluation_goal
        else:
            logger.info(
                f"Unknown property type: {original_property_str}. Using qvalues computed with given original property")
            qvalues = original_qvalues
        if self.rl_args.normalize_simulator_rewards:
            qvalues = qvalues / self.rl_args.evaluation_goal
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

    def run_rl_synthesis(self, saynt: bool = True):
        # assignment = self.storm_control.latest_paynt_result
        # qvalues = self.storm_control.qvalues
        fsc = self.storm_control.latest_paynt_result_fsc
        rl_synthesiser = Synthesizer_RL(self.quotient.pomdp, self.rl_args)
        if saynt:
            rl_synthesiser.get_saynt_trajectories(
                self.storm_control, self.quotient, fsc)
            return
        first_time = True
        repeated_fsc = False
        soft_decision = False
        logger.info("Training agent with combination of FSC and RL.")
        rl_synthesiser.train_agent_combined_with_fsc_advanced(
            1000, fsc, self.storm_control.paynt_bounds)
        return
        while True:
            logger.info("Training agent with FSC.")
            if fsc and (first_time or repeated_fsc):
                rl_synthesiser.train_agent_with_fsc_data(
                    100, fsc, soft_decision=soft_decision)
                first_time = False
            if soft_decision:
                rl_synthesiser.update_fsc_multiplier(0.5)
            logger.info("Training agent for {} iterations.".format(2000))
            rl_synthesiser.train_agent(2000)
            if not repeated_fsc:
                break
        rl_synthesiser.save_to_json("PAYNTc+RL")

    def run_rl_synthesis_critic(self):
        qvalues = self.storm_control.qvalues
        rl_synthesiser = Synthesizer_RL(
            self.quotient.pomdp, self.rl_args, qvalues=qvalues,
            action_labels_at_observation=self.quotient.action_labels_at_observation)
        rl_synthesiser.train_agent(2000)
        rl_synthesiser.save_to_json("PAYNTc_Critic+RL")

    def run_rl_synthesis_q_vals_rand(self):
        qvalues = self.storm_control.qvalues
        rl_synthesizer = Synthesizer_RL(
            self.quotient.pomdp, self.rl_args, qvalues=qvalues,
            action_labels_at_observation=self.quotient.action_labels_at_observation,
            random_init_starts_q_vals=True
        )
        rl_synthesizer.train_agent_qval_randomization(2000, qvalues)
        rl_synthesizer.save_to_json("PAYNTq_randomization")

    def property_is_reachability(self, property: str):
        return "Pmax" in property or "Pmin" in property

    def property_is_maximizing(self, property: str):
        return "max" in property

    def run_rl_synthesis_dqn_ppo(self, fsc_json_dict={}):
        fsc = self.storm_control.latest_paynt_result_fsc
        sub_method = self.input_rl_settings_dict["sub_method"]
        # self.quotient.get_property().__str__()
        original_property = fsc_json_dict["specification_property"]
        is_probab_condition = self.property_is_reachability(original_property)
        is_maximizing = self.property_is_maximizing(original_property)
        rl_synthesiser = Synthesizer_RL(
            self.quotient.pomdp, self.rl_args, pretrain_dqn=True)
        paynt_value = fsc_json_dict["paynt_value"]
        # = self.storm_control.paynt_bounds # TODO: SAYNT controller has the value defined in self.storm_control.storm_bounds
        rl_synthesiser.dqn_and_ppo_training(fsc, sub_method=sub_method, fsc_quality=paynt_value,
                                            maximizing_value=is_maximizing,
                                            probability_cond=is_probab_condition)
        rl_synthesiser.save_to_json(experiment_name=self.input_rl_settings_dict["agent_task"],
                                    model=self.input_rl_settings_dict["model_name"],
                                    method=f"{self.rl_args.learning_method}_{sub_method}")

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

    def iterative_storm_loop(self, timeout, paynt_timeout, storm_timeout, iteration_limit=0):
        self.run_rl = True
        self.combo_mode = RL_SAYNT_Combo_Modes.TRAJECTORY_MODE
        self.saynt = True
        self.try_faster = False
        self.rl_args = init_rl_args(mode=self.combo_mode)
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
            if self.try_faster and hasattr(self, "input_rl_settings_dict"):
                specification_property = self.quotient.get_property().__str__()
                paynt_value = self.storm_control.paynt_bounds
                storm_value = self.storm_control.storm_bounds
                fsc_json_dict = {"specification_property": specification_property,
                                 "paynt_value": paynt_value, "storm_value": storm_value}
                with open(f"{agent_task}/{fsc_file_name}", "wb") as f:
                    pickle.dump(self.storm_control.latest_paynt_result_fsc, f)
                with open(f"{agent_task}/{fsc_file_name}_info.json", "w") as f:
                    json.dump(fsc_json_dict, f)
        if self.run_rl and self.combo_mode == RL_SAYNT_Combo_Modes.TRAJECTORY_MODE:
            self.run_rl_synthesis(self.saynt)
        elif self.run_rl and self.combo_mode == RL_SAYNT_Combo_Modes.QVALUES_CRITIC_MODE:
            self.run_rl_synthesis_critic()
        elif self.run_rl and self.combo_mode == RL_SAYNT_Combo_Modes.QVALUES_RANDOM_SIM_INIT_MODE:
            self.run_rl_synthesis_q_vals_rand()
        elif self.run_rl and self.combo_mode == RL_SAYNT_Combo_Modes.DQN_AS_QTABLE:
            self.run_rl_synthesis_dqn_ppo(fsc_json_dict)

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

    def set_advices_from_rl(self, interpretation_result=None, load_rl_dict=True, rl_dict=False, family=None, rl_load_path=None):
        """ Set advices from RL interpretation.

        Args:
            interpretation_result (tuple, optional): Tuple of dictionaries (obs_act_dict, memory_dict, labels). Defaults to None.
            load_rl_dict (bool, optional): Whether to load RL interpretation from a file. Defaults to True.
            rl_dict (bool, optional): Whether to use RL interpretation. Defaults to False.
            family (paynt.quotient.quotient.QuotientFamily, optional): Family of the POMDP. Defaults to None.
        """
        if load_rl_dict:
            path_obs_act_dict = os.path.join(
                rl_load_path, "obs_action_dict.pickle")
            path_labels = os.path.join(rl_load_path, "labels.pickle")

            with open(path_obs_act_dict, "rb") as f:
                obs_actions = pickle.load(f)
            with open(path_labels, "rb") as f:
                action_keywords = pickle.load(f)
            obs_actions = self.storm_control.convert_rl_dict_to_paynt(
                family, obs_actions, action_keywords)
            self.storm_control.result_dict = obs_actions
            self.storm_control.result_dict_no_cutoffs = obs_actions
        elif rl_dict:
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

    # PAYNT POMDP synthesis that uses pre-computed results from Storm as guide

    def strategy_storm(self, unfold_imperfect_only, unfold_storm=True):
        '''
        @param unfold_imperfect_only if True, only imperfect observations will be unfolded
        '''
        mem_size = paynt.quotient.pomdp.PomdpQuotient.initial_memory_size
        self.synthesizer.storm_control = self.storm_control

        if hasattr(self, 'input_rl_settings_dict'):
            rl_dict = True
            fsc_cycling = self.input_rl_settings_dict['fsc_cycling']
            fsc_synthesis_time_limit = self.input_rl_settings_dict['fsc_synthesis_time_limit']
            load_rl_dict = self.input_rl_settings_dict['load_agent']
            soft_fsc = self.input_rl_settings_dict['soft_fsc']
            rl_pretrain_iters = self.input_rl_settings_dict['rl_pretrain_iters']
            rl_training_iters = self.input_rl_settings_dict['rl_training_iters']
            fsc_training_iterations = self.input_rl_settings_dict['fsc_training_iterations']
            rl_fsc_multiplier = self.input_rl_settings_dict['fsc_multiplier']
            rl_load_path = self.input_rl_settings_dict['rl_load_path']
            rl_load_memory_flag = self.input_rl_settings_dict['rl_load_memory_flag']
        else:  # Set all reinforcement learning settings to False or 0
            rl_dict = False
            fsc_cycling = False
            fsc_synthesis_time_limit = 0
            load_rl_dict = False
            soft_fsc = False
            rl_pretrain_iters = 0
            rl_training_iters = 0
            fsc_training_iterations = 0
            rl_fsc_multiplier = 0
            rl_load_path = None
            rl_load_memory_flag = False

        first_run = True
        current_time = None
        if fsc_cycling:
            current_time = fsc_synthesis_time_limit
        interpretation_result = None

        if rl_dict and not load_rl_dict:
            args = ArgsEmulator(load_agent=False, learning_method="PPO",
                                encoding_method="Valuations", max_steps=300, restart_weights=0, agent_name="PAYNT")
            rl_synthesiser = Synthesizer_RL(
                self.quotient.pomdp, args, fsc_pre_init=soft_fsc)
            rl_synthesiser.train_agent(rl_pretrain_iters)
            interpretation_result = rl_synthesiser.interpret_agent(best=False)

        while True:
            if rl_dict and fsc_cycling and not first_run:
                current_time = fsc_synthesis_time_limit
                if fsc is not None:
                    logger.info("Training agent with FSC.")
                    rl_synthesiser.train_agent_with_fsc_data(
                        fsc_training_iterations, fsc, soft_decision=soft_fsc)
                else:
                    logger.info("FSC is None. Training agent without FSC.")
                logger.info("Training agent for {} iterations.".format(
                    rl_training_iters))
                rl_synthesiser.train_agent(rl_training_iters)
                interpretation_result = rl_synthesiser.interpret_agent(
                    best=False)

            if self.storm_control.is_storm_better == False:
                self.storm_control.parse_results(self.quotient)

            paynt.quotient.pomdp.PomdpQuotient.current_family_index = mem_size
            if interpretation_result is not None:
                self.memory_dict = interpretation_result[1]
                self.priority_list = interpretation_result[3]

            if load_rl_dict and rl_load_memory_flag:
                path_memory_dict = os.path.join(
                    rl_load_path, "memory_dict.pickle")
                with open(path_memory_dict, "rb") as f:
                    obs_memory_dict = pickle.load(f)
                    for key in obs_memory_dict.keys():
                        if obs_memory_dict[key] <= 0:
                            obs_memory_dict[key] = 1
                    self.quotient.set_memory_from_dict(obs_memory_dict)

            # unfold memory according to the best result
            if (not rl_dict or interpretation_result is None) and unfold_storm:
                self.set_memory_original(mem_size)
            elif rl_dict and not first_run:
                logger.info("Adding memory nodes based on RL interpretation.")

                priority_list_len = int(
                    math.ceil(len(self.priority_list) / 10))
                priorities = self.priority_list[:priority_list_len]
                for i in range(len(priorities)):
                    self.memory_dict[priorities[i]] += 1
                self.quotient.set_memory_from_dict(self.memory_dict)
            else:
                logger.info(
                    "Synthesizing optimal k={} controller ...".format(mem_size))
                if unfold_imperfect_only:
                    self.quotient.set_imperfect_memory_size(mem_size)
                else:
                    self.quotient.set_global_memory_size(mem_size)

            family = self.quotient.family

            if load_rl_dict or (rl_dict and interpretation_result is not None):
                self.set_advices_from_rl(interpretation_result=interpretation_result, load_rl_dict=load_rl_dict, rl_dict=rl_dict,
                                         family=family, rl_load_path=rl_load_path)

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

            if fsc_cycling:
                assignment = self.synthesize(family, timer=current_time)
                try:
                    fsc = self.quotient.assignment_to_fsc(assignment)
                    rl_synthesiser.update_fsc_multiplier(rl_fsc_multiplier)
                except:
                    logger.info(
                        "FSC could not be created from the assignment. Probably no improvement.")
                    rl_synthesiser.update_fsc_multiplier(1 / rl_fsc_multiplier)
            else:
                assignment = self.synthesize(family)

            if assignment is not None:
                self.storm_control.latest_paynt_result = assignment
                self.storm_control.paynt_export = self.quotient.extract_policy(
                    assignment)
                self.storm_control.paynt_bounds = self.quotient.specification.optimality.optimum

            first_run = False
            self.storm_control.update_data()

            mem_size += 1

            # break

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
