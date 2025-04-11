from rl_src.tools.evaluation_results_class import EvaluationResults
from rl_src.interpreters.fsc_based_interpreter import NaiveFSCPolicyExtraction
from paynt.family.family import Family
from paynt.rl_extension.family_extractors.external_family_wrapper import ExtractedFamilyWrapper

from rl_src.experimental_interface import ArgsEmulator
from rl_src.interpreters.bottlenecking.quantized_bottleneck_extractor import BottleneckExtractor
from rl_src.tools.evaluators import evaluate_policy_in_model
from rl_src.interpreters.direct_fsc_extraction.direct_extractor import *

from .synthesizer_onebyone import SynthesizerOneByOne
from paynt.rl_extension.saynt_rl_tools.agents_wrapper import AgentsWrapper

from paynt.rl_extension.saynt_rl_tools.rl_saynt_combo_modes import RL_SAYNT_Combo_Modes, init_rl_args

from paynt.quotient.storm_pomdp_control import StormPOMDPControl
from paynt.quotient.pomdp import PomdpQuotient

from rl_src.tools.specification_check import SpecificationChecker

import time

import tensorflow as tf

import logging
logger = logging.getLogger(__name__)

from rl_src.interpreters.direct_fsc_extraction.direct_extractor import DirectExtractor
from rl_src.interpreters.direct_fsc_extraction.extraction_stats import ExtractionStats
from paynt.rl_extension.extraction_benchmark_res import ExtractionBenchmarkRes, ExtractionBenchmarkResManager

class SynthesizerRL:
    def __init__(self, quotient: PomdpQuotient, method: str, storm_control: StormPOMDPControl, input_rl_settings: dict = None,
                 use_one_hot_memory = False):
        self.quotient = quotient
        self.use_storm = False
        self.synthesizer = None
        self.set_input_rl_settings_for_paynt(input_rl_settings)
        self.synthesizer = SynthesizerOneByOne
        self.total_iters = 0

        if storm_control is not None:
            self.use_storm = True
            self.storm_control = storm_control
        self.use_one_hot_memory = use_one_hot_memory

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
        agent_name = "SAYNT_Booster"
        rnn_less = self.rnn_less
        args = ArgsEmulator(learning_rate=1.6e-4,
                            restart_weights=0, learning_method="Stochastic_PPO", prism_model=f"fake_path/{self.model_name}/sketch.templ",
                            nr_runs=nr_runs, agent_name=agent_name, load_agent=False,
                            evaluate_random_policy=False, max_steps=1501, evaluation_goal=50.0, evaluation_antigoal=-0.0,
                            trajectory_num_steps=32, discount_factor=0.99, num_environments=256,
                            normalize_simulator_rewards=False, buffer_size=200, random_start_simulator=False,
                            batch_size=256, vectorized_envs_flag=True, perform_interpretation=False,
                            use_rnn_less=rnn_less, model_memory_size=0, state_supporting=False,
                            completely_greedy=False, prefer_stochastic=False, model_name=self.model_name)
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
        self.model_name = input_rl_settings_dict["model_name"]
        self.args = self.create_rl_args(input_rl_settings_dict)
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

    def bottleneck_extraction(self, agents_wrapper : AgentsWrapper, 
                              input_dim : int =64, latent_dim : int = 1, 
                              best_extractor : BottleneckExtractor = None, 
                              best_result : EvaluationResults = None
                              ) -> tuple[BottleneckExtractor, EvaluationResults]:
        bottleneck_extractor = BottleneckExtractor(
                agents_wrapper.agent.tf_environment, input_dim, latent_dim=latent_dim)
        bottleneck_extractor.train_autoencoder(
                agents_wrapper.agent.wrapper, num_epochs=501, num_data_steps=self.args.max_steps + 1)
        evaluation_result = bottleneck_extractor.evaluate_bottlenecking(
                agents_wrapper.agent, max_steps=self.args.max_steps + 1)
        if best_extractor is None or evaluation_result.best_reach_prob > best_result.best_reach_prob:
                best_extractor = bottleneck_extractor
                best_result = evaluation_result
        elif evaluation_result.best_reach_prob == best_result.best_reach_prob and evaluation_result.best_return > best_result.best_return:
                best_extractor = bottleneck_extractor
                best_result = evaluation_result
        return best_extractor, best_result

    def perform_bottleneck_extraction(self, agents_wrapper: AgentsWrapper):
        input_dim = 64
        latent_dim = 1
        best_bottleneck_extractor = None
        best_evaluation_result = None
        for i in range(2):
            best_bottleneck_extractor, best_evaluation_result = self.bottleneck_extraction(
                agents_wrapper, input_dim, latent_dim, best_bottleneck_extractor, best_evaluation_result)
        bottleneck_extractor = best_bottleneck_extractor
        agents_wrapper.agent.wrapper.set_greedy(True)
        extracted_fsc = bottleneck_extractor.extract_fsc(
            policy=agents_wrapper.agent.wrapper, environment=agents_wrapper.agent.environment)
        evaluation_result = evaluate_policy_in_model(
            extracted_fsc, agents_wrapper.agent.args, agents_wrapper.agent.environment, agents_wrapper.agent.tf_environment)
        agents_wrapper.agent.wrapper.set_greedy(False)
        return bottleneck_extractor, extracted_fsc, evaluation_result, latent_dim

    def perform_rl_to_fsc_cloning(self, policy : TFPolicy, 
                                  environment : EnvironmentWrapperVec, 
                                  tf_environment : TFPyEnvironment, latent_dim=2):
        if "Pmax" in self.quotient.get_property().__str__():
            optimization_specification = SpecificationChecker.Constants.REACHABILITY
        else:
            optimization_specification = SpecificationChecker.Constants.REWARD

        direct_extractor = DirectExtractor(memory_len = latent_dim, is_one_hot=self.use_one_hot_memory,
                                           use_residual_connection=True, training_epochs=30001,
                                           num_data_steps=(self.args.max_steps + 1) * 6, get_best_policy_flag=False, model_name=self.model_name,
                                           max_episode_len=self.args.max_steps, optimizing_specification=optimization_specification)
        fsc, extraction_stats = direct_extractor.clone_and_generate_fsc_from_policy(
            policy, environment, tf_environment)
        extraction_stats.store_as_json(self.model_name, "experiments_loopy_fscs")
        return fsc, extraction_stats
    
    # def perform_policy_to_fsc_cloning(self, policy : TFPolicy, 
    #                                   environment : EnvironmentWrapperVec, 
    #                                   tf_environment : TFPyEnvironment, latent_dim=2):
    #     buffer = sample_data_with_policy(policy, 5000, environment, tf_environment)
    #     cloned_fsc_actor = behavioral_clone_original_policy_to_fsc(
    #         buffer, 70000, policy, latent_dim, sample_len=self.args.trajectory_num_steps, 
    #         observation_and_action_constraint_splitter=self.agents_wrapper.agent.wrapper.observation_and_action_constraint_splitter,
    #         environment=self.agents_wrapper.agent.environment,
    #         tf_environment=self.agents_wrapper.agent.tf_environment,
    #         args=self.args)
    #     fsc = extract_fsc(
    #         cloned_fsc_actor, environment, latent_dim, self.args)
    #     return fsc

    def single_shot_synthesis(self, agents_wrapper: AgentsWrapper, nr_rl_iterations: int, paynt_timeout: int, fsc=None, storm_control=None,
                              bottlenecking = False):

        if storm_control is not None:
            trajectories = agents_wrapper.generate_saynt_trajectories(
                storm_control, self.quotient, fsc=fsc, model_reward_multiplier=agents_wrapper.agent.environment.reward_multiplier,
                tf_action_labels=agents_wrapper.agent.environment.action_keywords, num_episodes=32)
            agents_wrapper.train_with_bc(
                nr_of_iterations=nr_rl_iterations // 20, trajectories=trajectories)
            agents_wrapper.train_agent(nr_rl_iterations)
        elif fsc is not None:
            if self.combo_mode == RL_SAYNT_Combo_Modes.JUMPSTART_MODE:
                self.run_rl_synthesis_jumpstarts(
                    fsc, saynt=self.saynt, save=False, nr_of_iterations=nr_rl_iterations)
            elif self.combo_mode == RL_SAYNT_Combo_Modes.SHAPING_MODE:
                self.run_rl_synthesis_shaping(
                    fsc, saynt=self.saynt, save=False, nr_of_iterations=nr_rl_iterations)
            elif self.combo_mode == RL_SAYNT_Combo_Modes.BEHAVIORAL_CLONING:
                agents_wrapper.train_with_bc(
                    fsc, nr_of_iterations=nr_rl_iterations)
            else:
                logger.error(
                    "Not implemented combo mode, running baseline training.")
                agents_wrapper.train_agent(nr_rl_iterations)
        else:
            agents_wrapper.train_agent(nr_rl_iterations)

        # TODO: Explore option, where the extraction is performed the best agent with agents_wrapper.agent.load_agent(True)
        agents_wrapper.agent.set_agent_greedy()
        latent_dim = 2 if not self.use_one_hot_memory else 5
        if bottlenecking:
            bottleneck_extractor, extracted_fsc, _, latent_dim = self.perform_bottleneck_extraction(
                agents_wrapper)
        else:
            
            extracted_fsc, _ = self.perform_rl_to_fsc_cloning(
                agents_wrapper.agent.wrapper, 
                agents_wrapper.agent.environment, 
                agents_wrapper.agent.tf_environment, 
                latent_dim=latent_dim)
        agents_wrapper.agent.set_agent_stochastic()
        assignment = self.compute_paynt_assignment_from_fsc_like(extracted_fsc, latent_dim=latent_dim, agents_wrapper=agents_wrapper)
        return assignment
    
    def compute_paynt_assignment_from_fsc_like(self, fsc_like, latent_dim=2, agents_wrapper=None, paynt_timeout=60):

        fsc_size = 3 ** latent_dim if not self.use_one_hot_memory else latent_dim
        self.quotient.set_imperfect_memory_size(fsc_size)

        family = self.quotient.family

        initialized_extraction = ExtractedFamilyWrapper(
            family, 0, agents_wrapper, greedy=self.greedy, memory_only=self.memory_only_subfamilies,
            extracted_bottlenecked_fsc=fsc_like)

        main_family = initialized_extraction.get_family()
        assignment = self.synthesize(
            main_family, timer=paynt_timeout, print_stats=True)
        alternative_assignment = self.synthesize(
            family=family, timer=paynt_timeout, print_stats=True)
        if alternative_assignment is None:
            logger.info("No improving assignment found.")
        else:
            assignment = alternative_assignment
        return assignment
    
    def perform_benchmarking(self, agents_wrapper: AgentsWrapper, number_of_runs=10):
        methods = ["Bottlenecking", "Direct_Tanh", "Direct_OneHot"]
        benchmark_results = []
        sizes_bottlenecking = [1, 2]
        sizes_direct_tanh = [1, 2]
        sizes_direct_onehot = [3, 5, 9]
        agents_wrapper.train_agent(2001) # We train only a single network
        original_rl_reward = agents_wrapper.agent.evaluation_result.returns[-1]
        original_rl_reachability = agents_wrapper.agent.evaluation_result.reach_probs[-1]
        agents_wrapper.agent.set_agent_greedy()
        method_sizes_map = {
            "Bottlenecking": sizes_bottlenecking,
            "Direct_Tanh": sizes_direct_tanh,
            "Direct_OneHot": sizes_direct_onehot
        }
        # agents_wrapper.agent.load_agent(True)
        
        for i in range(number_of_runs):
            for method, sizes in method_sizes_map.items():
                for size in sizes:
                    if method == "Bottlenecking":
                        self.use_one_hot_memory = False
                        bottleneck_extractor = BottleneckExtractor(
                            agents_wrapper.agent.tf_environment, input_dim=64, latent_dim=size)
                        bottleneck_extractor.train_autoencoder(
                            agents_wrapper.agent.wrapper, num_epochs=80, num_data_steps=(self.args.max_steps + 1) * 6)
                        extracted_fsc = bottleneck_extractor.extract_fsc(
                            policy=agents_wrapper.agent.wrapper, environment=agents_wrapper.agent.environment)
                        assignment = self.compute_paynt_assignment_from_fsc_like(
                            extracted_fsc, latent_dim=size, agents_wrapper=agents_wrapper)
                        logger.info(
                            f"Benchmarking {method} with size {size} finished. Verified performance: {self.quotient.specification.optimality.optimum}.")
                        # paynt_export = self.quotient.extract_policy(assignment)
                        paynt_bounds = self.quotient.specification.optimality.optimum
                        benchmark_result = ExtractionBenchmarkRes(
                            type=method, memory_size=size, accuracies=[0.0], verified_performance=paynt_bounds,
                            original_rl_reward=original_rl_reward, original_rl_reachability=original_rl_reachability,
                            reachabilities=[0.0], rewards=[0.0])
                    else:  # Direct_Tanh or Direct_OneHot
                        self.use_one_hot_memory = True if method == "Direct_OneHot" else False
                        extracted_fsc, stats = self.perform_rl_to_fsc_cloning(
                            agents_wrapper.agent.wrapper, agents_wrapper.agent.environment, agents_wrapper.agent.tf_environment, latent_dim=size)

                        assignment = self.compute_paynt_assignment_from_fsc_like(
                        extracted_fsc, latent_dim=size, agents_wrapper=agents_wrapper)
                        logger.info(f"Exporting assignment.")
                        # paynt_export = self.quotient.extract_policy(assignment)
                        paynt_bounds = self.quotient.specification.optimality.optimum

                        benchmark_result = ExtractionBenchmarkRes(
                            type=method, memory_size=size, accuracies=stats.evaluation_accuracies, verified_performance=paynt_bounds,
                            original_rl_reward=original_rl_reward, original_rl_reachability=original_rl_reachability,
                            reachabilities=stats.extracted_policy_reachabilities, rewards=stats.extracted_policy_rewards)
                    benchmark_results.append(benchmark_result)
                    self.quotient.specification.reset()
                    logger.info(
                        f"Benchmarking {method} with size {size} finished. Verified performance: {paynt_bounds}.")
                    ExtractionBenchmarkResManager.create_folder_with_extraction_benchmark_res(
                        f"experiments_extraction/{self.model_name}", benchmark_results)
        return benchmark_results
                    
                


    def run(self, multiple_assignments_benchmark = False, bottlenecking = False):
        if not hasattr(self.quotient, "pomdp"):
            pomdp = self.quotient.quotient_mdp
        else:
            pomdp = self.quotient.pomdp
        agents_wrapper = AgentsWrapper(
            pomdp, self.args, agent_folder=self.model_name)
        self.set_agents_wrapper(agents_wrapper)
        start_time = time.time()
        if multiple_assignments_benchmark:
            benchmark_results = self.perform_benchmarking(agents_wrapper)
            ExtractionBenchmarkResManager.create_folder_with_extraction_benchmark_res(
                f"experiments_extraction/{self.model_name}", benchmark_results)
            return None
        fsc = None
        while True:
            assignment = self.single_shot_synthesis(
                agents_wrapper, self.rl_training_iters, self.fsc_synthesis_time_limit, fsc,
                bottlenecking=bottlenecking)
            if assignment is not None:
                agents_wrapper.agent.evaluation_result.add_paynt_bound(
                    self.quotient.specification.optimality.optimum)
                fsc = self.quotient.assignment_to_fsc(assignment)

            if not self.loop:
                break
            if time.time() - start_time > self.time_limit:
                break
        # Save the final json file
        agents_wrapper.save_to_json(
            self.args.agent_name, model=self.model_name, method=self.args.learning_method)
        return assignment

    def finalize_synthesis(self, assignment):
        if assignment is not None:
            self.storm_control.latest_paynt_result = assignment
            # print(assignment)
            self.storm_control.paynt_export = self.quotient.extract_policy(
                assignment)
            self.storm_control.paynt_bounds = self.quotient.specification.optimality.optimum
            self.storm_control.paynt_fsc_size = self.quotient.policy_size(
                self.storm_control.latest_paynt_result)
            # self.storm_control.latest_paynt_result_fsc = self.quotient.assignment_to_fsc(
            #     self.storm_control.latest_paynt_result)
            # self.storm_control.qvalues = self.compute_qvalues_for_rl(
            #     assignment=assignment)
        else:
            logging.info("Assignment is None")

        self.storm_control.update_data()

    def set_agents_wrapper(self, agents_wrapper: AgentsWrapper):
        self.agents_wrapper = agents_wrapper

    def get_agents_wrapper(self) -> AgentsWrapper:
        if not hasattr(self, "agents_wrapper") or self.agents_wrapper is None:
            self.agents_wrapper = AgentsWrapper(
                self.quotient.pomdp, self.args, agent_folder=self.model_name)
        return self.agents_wrapper
