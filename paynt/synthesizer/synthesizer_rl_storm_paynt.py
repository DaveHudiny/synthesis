from rl_src.tools.evaluators import evaluate_extracted_fsc
from rl_src.interpreters.fsc_based_interpreter import ExtractedFSCPolicy
from paynt.family.family import Family
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

from ..rl_extension.saynt_rl_tools.rl_saynt_combo_modes import RL_SAYNT_Combo_Modes, init_rl_args

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
        nr_runs = 4001
        agent_name = "RevisitedLoop"
        rnn_less = True
        args = ArgsEmulator(learning_rate=1.6e-4,
                            restart_weights=0, learning_method="Stochastic_PPO",
                            nr_runs=nr_runs, agent_name=agent_name, load_agent=False,
                            evaluate_random_policy=False, max_steps=400, evaluation_goal=50, evaluation_antigoal=-20,
                            trajectory_num_steps=32, discount_factor=0.99, num_environments=256,
                            normalize_simulator_rewards=False, buffer_size=500, random_start_simulator=True,
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
        else:
            self.fsc_synthesis_time_limit = input_rl_settings_dict['time_limit']
        # if self.loop:
        self.time_limit = input_rl_settings_dict['time_limit']
        self.rnn_less = input_rl_settings_dict['rnn_less']
        self.args = self.create_rl_args(input_rl_settings_dict)

    # SAYNT reimplementation with RL. Returns the extracted main family and subfamilies.
    def process_rl_hint(self, rl_agent) -> tuple[Family, list[Family]]:
        restricted_main_family = self.quotient.family.copy()
        subfamily_restrictions = []
        return restricted_main_family, subfamily_restrictions

    def process_storm_hint(self) -> tuple[Family, list[Family]]:
        restricted_main_family = self.quotient.family.copy()
        subfamily_restrictions = []
        return restricted_main_family, subfamily_restrictions

    def extract_one_fsc_w_entropy(self, agents_wrapper: AgentsWrapper) -> ExtractedFSCPolicy:
        extracted_fsc = ExtractedFSCPolicy(agents_wrapper.agent.wrapper, agents_wrapper.agent.environment,
                                           agents_wrapper.agent.tf_environment, self.args, entropy_extraction=True)
        return extracted_fsc
    
    def set_memory_w_extracted_fsc_entropy(self, extracted_fsc : ExtractedFSCPolicy, ceil=True):
        obs_memory_dict = {}
        bit_entropies = extracted_fsc.observation_to_entropy_table
        memory_entropies = tf.pow(2.0, bit_entropies)
        if ceil:
            memory_entropies = tf.math.ceil(memory_entropies).numpy()
        else:
            memory_entropies = tf.math.floor(memory_entropies).numpy()
        memory_entropies = memory_entropies.astype(int)
        for observation in range(self.quotient.pomdp.nr_observations):
            obs_memory_dict[observation] = memory_entropies[observation]
        print(obs_memory_dict)
        self.quotient.set_memory_from_dict(obs_memory_dict)

    def initialize_main_family_w_extracted_fsc(self, extracted_fsc) -> Family:
        pass

    def iterative_storm_rl_paynt(self, timeout, paynt_timeout, storm_timeout, iteration_limit=0):
        pass

    def run(self):
        agents_wrapper = AgentsWrapper(self.quotient.pomdp, self.args)
        agents_wrapper.train_agent(1000)
        extracted_fsc = self.extract_one_fsc_w_entropy(agents_wrapper)
        self.set_memory_w_extracted_fsc_entropy(extracted_fsc)

        family = self.quotient.family

        initialized_extraction = ExtractedFamilyWrapper(family, 0, agents_wrapper)
        subfamily_restrictions = initialized_extraction.get_subfamily_restrictions()
        subfamilies = self.storm_control.get_subfamilies(
            subfamily_restrictions, family)
        self.synthesizer.subfamilies_buffer = subfamilies
        self.synthesizer.main_family = initialized_extraction.get_family()
        assignment = self.synthesize(family, timer=600)
        self.finalize_synthesis(assignment)
        
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