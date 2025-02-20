from rl_src.tools.evaluators import evaluate_extracted_fsc
from rl_src.interpreters.fsc_based_interpreter import NaiveFSCPolicyExtraction
from paynt.family.family import Family
import json
import numpy as np
from time import sleep
import os
from paynt.rl_extension.family_extractors.external_family_wrapper import ExtractedFamilyWrapper

from rl_src.experimental_interface import ArgsEmulator
from rl_src.interpreters.bottlenecking.quantized_bottleneck_extractor import BottleneckExtractor, TableBasedPolicy
from rl_src.tools.evaluators import evaluate_policy_in_model

import pickle
import stormpy

from .statistic import Statistic
import paynt.synthesizer.synthesizer_ar
from .synthesizer_ar_storm import SynthesizerARStorm
from .synthesizer_hybrid import SynthesizerHybrid
from .synthesizer_multicore_ar import SynthesizerMultiCoreAR
from .synthesizer_onebyone import SynthesizerOneByOne
from paynt.rl_extension.saynt_rl_tools.agents_wrapper import AgentsWrapper

from paynt.rl_extension.saynt_rl_tools.rl_saynt_combo_modes import RL_SAYNT_Combo_Modes, init_rl_args

from paynt.rl_extension.saynt_rl_tools.regex_patterns import RegexPatterns
from paynt.rl_extension.family_extractors.external_family_wrapper import RLFamilyExtractor
from paynt.parser.prism_parser import PrismParser
from paynt.quotient.storm_pomdp_control import StormPOMDPControl
from paynt.quotient.pomdp import PomdpQuotient

import paynt.quotient.quotient
import paynt.quotient.pomdp
import paynt.utils.timer

import paynt.verification.property

import math
from collections import defaultdict

from threading import Thread
from queue import Queue
import time

import tensorflow as tf

import logging
logger = logging.getLogger(__name__)


class SynthesizerRL:
    def __init__(self, quotient: PomdpQuotient, method: str, storm_control: StormPOMDPControl, input_rl_settings: dict = None):
        self.quotient = quotient
        self.use_storm = False
        self.synthesizer = None
        self.set_input_rl_settings_for_paynt(input_rl_settings)
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
            self.synthesizer = SynthesizerOneByOne
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

    def set_combo_setting(self, input_rl_settings_dict: dict):
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

    def create_rl_args(self, input_rl_settings_dict: dict):
        nr_runs = self.rl_training_iters
        agent_name = "RevisitedLoop"
        rnn_less = self.rnn_less
        args = ArgsEmulator(learning_rate=1.6e-4,
                            restart_weights=0, learning_method="Stochastic_PPO",
                            nr_runs=nr_runs, agent_name=agent_name, load_agent=False,
                            evaluate_random_policy=False, max_steps=400, evaluation_goal=50, evaluation_antigoal=-20,
                            trajectory_num_steps=30, discount_factor=0.99, num_environments=256,
                            normalize_simulator_rewards=False, buffer_size=500, random_start_simulator=False,
                            batch_size=256, vectorized_envs_flag=True, perform_interpretation=True, use_rnn_less=rnn_less, model_memory_size=0)
        return args

    def set_input_rl_settings_for_paynt(self, input_rl_settings_dict):
        self.rl_settings = input_rl_settings_dict
        self.set_combo_setting(input_rl_settings_dict)
        self.use_rl = True
        self.loop = input_rl_settings_dict["loop"]
        self.rl_training_iters = input_rl_settings_dict['rl_training_iters']
        self.rl_load_memory_flag = input_rl_settings_dict['rl_load_memory_flag']
        self.greedy = input_rl_settings_dict['greedy']
        if self.loop:
            self.fsc_synthesis_time_limit = input_rl_settings_dict['fsc_time_in_loop']
            self.time_limit = input_rl_settings_dict['time_limit']
        else:
            self.fsc_synthesis_time_limit = input_rl_settings_dict['time_limit']
            self.time_limit = input_rl_settings_dict['time_limit']

        # if self.loop:
        self.time_limit = input_rl_settings_dict['time_limit']
        self.rnn_less = input_rl_settings_dict['rnn_less']
        self.args = self.create_rl_args(input_rl_settings_dict)
        self.model_name = input_rl_settings_dict["model_name"]
        self.memory_only_subfamilies = False

    # SAYNT reimplementation with RL. Returns the extracted main family and subfamilies.

    def process_rl_hint(self, rl_agent) -> tuple[Family, list[Family]]:
        restricted_main_family = self.quotient.family.copy()
        subfamily_restrictions = []
        return restricted_main_family, subfamily_restrictions

    def process_storm_hint(self) -> tuple[Family, list[Family]]:
        restricted_main_family = self.quotient.family.copy()
        subfamily_restrictions = []
        return restricted_main_family, subfamily_restrictions

    def extract_one_fsc_w_entropy(self, agents_wrapper: AgentsWrapper, greedy: bool = False) -> NaiveFSCPolicyExtraction:
        self.extracted_fsc = NaiveFSCPolicyExtraction(agents_wrapper.agent.wrapper, agents_wrapper.agent.environment,
                                           agents_wrapper.agent.tf_environment, self.args, entropy_extraction=True,
                                           greedy=greedy, max_memory_size=4)
        
        return self.extracted_fsc

    def set_memory_w_extracted_fsc_entropy(self, extracted_fsc: NaiveFSCPolicyExtraction, ceil=True):
        obs_memory_dict = {}
        bit_entropies = extracted_fsc.observation_to_entropy_table
        memory_entropies = tf.pow(2.0, bit_entropies)
        clipped_memory_entropies = tf.clip_by_value(memory_entropies, 1, 4)
        if ceil:
            memory_entropies = tf.math.ceil(clipped_memory_entropies).numpy()
        else:
            memory_entropies = tf.math.floor(clipped_memory_entropies).numpy()
        memory_entropies = memory_entropies.astype(int)
        for observation in range(self.quotient.pomdp.nr_observations):
            obs_memory_dict[observation] = memory_entropies[observation]
        self.quotient.set_memory_from_dict(obs_memory_dict)

    def initialize_main_family_w_extracted_fsc(self, extracted_fsc) -> Family:
        pass

    def iterative_storm_rl_paynt(self, timeout, paynt_timeout, storm_timeout, iteration_limit=0):
        pass

    def run_rl_synthesis_jumpstarts(self, fsc, saynt: bool = False, save=True, nr_of_iterations=4000):
        if saynt:
            raise NotImplementedError("SAYNT jumpstarts not implemented yet")
        agents_wrapper = self.get_agents_wrapper()
        agents_wrapper.train_agent_with_jumpstarts(fsc, nr_of_iterations)
        if save:
            agents_wrapper.save_to_json(
                self.args.agent_name, model=self.model_name, method=self.args.learning_method)

    def run_rl_synthesis_shaping(self, fsc, saynt: bool = False, save=True, nr_of_iterations=4000):
        if saynt:
            raise NotImplementedError("SAYNT shaping not implemented yet")
        agents_wrapper = self.get_agents_wrapper()
        agents_wrapper.train_agent_with_shaping(fsc, nr_of_iterations)
        experiment_name = f"{self.args.agent_name}_longer"
        if save:
            agents_wrapper.save_to_json(
                experiment_name, model=self.model_name, method=self.args.learning_method)

    def single_shot_synthesis(self, agents_wrapper: AgentsWrapper, nr_rl_iterations: int, paynt_timeout: int, fsc=None):
        if fsc is not None:
            if self.combo_mode == RL_SAYNT_Combo_Modes.JUMPSTART_MODE:
                self.run_rl_synthesis_jumpstarts(
                    fsc, saynt=self.saynt, save=False, nr_of_iterations=nr_rl_iterations)
            elif self.combo_mode == RL_SAYNT_Combo_Modes.SHAPING_MODE:
                self.run_rl_synthesis_shaping(
                    fsc, saynt=self.saynt, save=False, nr_of_iterations=nr_rl_iterations)
            else:
                logger.error(
                    "Not implemented combo mode, running baseline training.")
                agents_wrapper.train_agent(nr_rl_iterations)
        else:
            agents_wrapper.train_agent(nr_rl_iterations)
        agents_wrapper.agent.load_agent(True)
        if True: # Extract FSC via bottleneck extraction
            input_dim = 64
            latent_dim = 1
            best_bottleneck_extractor = None
            best_evaluation_result = None
            for i in range(5):
                bottleneck_extractor = BottleneckExtractor(agents_wrapper.agent.tf_environment, input_dim, latent_dim=latent_dim)
                bottleneck_extractor.train_autoencoder(agents_wrapper.agent.wrapper, num_epochs=50, num_data_steps=1000)
                evaluation_result = bottleneck_extractor.evaluate_bottlenecking(agents_wrapper.agent)
                if best_bottleneck_extractor is None or evaluation_result.best_reach_prob > best_evaluation_result.best_reach_prob:
                    best_bottleneck_extractor = bottleneck_extractor
                    best_evaluation_result = evaluation_result
            bottleneck_extractor = best_bottleneck_extractor
            extracted_fsc = bottleneck_extractor.extract_fsc(policy = agents_wrapper.agent.wrapper, environment = agents_wrapper.agent.environment)
            evaluation_result = evaluate_policy_in_model(extracted_fsc, agents_wrapper.agent.args, agents_wrapper.agent.environment, agents_wrapper.agent.tf_environment)
            print(f"Extracted FSC evaluation result: {evaluation_result}")
            self.quotient.set_global_memory_size(3 ** latent_dim)
        else:
            extracted_fsc = self.extract_one_fsc_w_entropy(
                agents_wrapper, greedy=self.greedy)
            self.set_memory_w_extracted_fsc_entropy(extracted_fsc)

        family = self.quotient.family
        
        if True:
            initialized_extraction = ExtractedFamilyWrapper(
                family, 0, agents_wrapper, greedy=self.greedy, memory_only=self.memory_only_subfamilies,
                extracted_bottlenecked_fsc=extracted_fsc)
        else:
            initialized_extraction = ExtractedFamilyWrapper(
                family, 0, agents_wrapper, greedy=self.greedy, memory_only=self.memory_only_subfamilies,
                extracted_fsc=self.extracted_fsc)
        # subfamily_restrictions = initialized_extraction.get_subfamily_restrictions()
        # subfamilies = self.storm_control.get_subfamilies(
        #     subfamily_restrictions, family)

        main_family = initialized_extraction.get_family()
        self.synthesizer.main_family = main_family
        del bottleneck_extractor
        del extracted_fsc
        # synthesizer = self.synthesizer(self.quotient)
        # synthesizer.stat = paynt.synthesizer.statistic.Statistic(self)
        # self.subfamilies_buffer = []
        # assignment = self.synthesize(family, timer=paynt_timeout)
        # self.synthesizer.subfamilies_buffer = subfamilies
        assignment = self.synthesize(main_family, timer=paynt_timeout)
        better_assignment = self.synthesize(self.quotient.family, timer=paynt_timeout)
        if better_assignment is not None:
            assignment = better_assignment
        self.finalize_synthesis(assignment)
        return assignment

    def run(self):
        agents_wrapper = AgentsWrapper(self.quotient.pomdp, self.args, agent_folder=self.model_name)
        self.set_agents_wrapper(agents_wrapper)
        start_time = time.time()
        fsc = None
        while True:
            assignment = self.single_shot_synthesis(
                agents_wrapper, self.rl_training_iters, self.fsc_synthesis_time_limit, fsc)
            if assignment is not None:
                agents_wrapper.agent.evaluation_result.add_paynt_bound(
                    self.storm_control.paynt_bounds)
                fsc = self.quotient.assignment_to_fsc(assignment)
            if not self.loop:
                break
            if time.time() - start_time > self.time_limit:
                break

        # Save the final json file
        agents_wrapper.save_to_json(
            self.args.agent_name, model=self.model_name, method=self.args.learning_method)

    def finalize_synthesis(self, assignment):
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

    def set_agents_wrapper(self, agents_wrapper: AgentsWrapper):
        self.agents_wrapper = agents_wrapper

    def get_agents_wrapper(self) -> AgentsWrapper:
        return self.agents_wrapper
