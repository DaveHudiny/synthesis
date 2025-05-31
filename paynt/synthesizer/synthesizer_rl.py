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
from rl_src.interpreters.aalpy_extraction.aalpy_extractor import AALpyExtractor
from paynt.rl_extension.extraction_benchmark_res import ExtractionBenchmarkRes, ExtractionBenchmarkResManager

from rl_src.interpreters.extracted_fsc.table_based_policy import TableBasedPolicy

from paynt.rl_extension.family_extractors.direct_fsc_construction import ConstructorFSC

import stormpy

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
                            restart_weights=0, learning_method="PPO", prism_model=f"fake_path/{self.model_name}/sketch.templ",
                            nr_runs=nr_runs, agent_name=agent_name, load_agent=False,
                            evaluate_random_policy=False, max_steps=401, evaluation_goal=50.0, evaluation_antigoal=-0.0,
                            trajectory_num_steps=32, discount_factor=0.99, num_environments=256,
                            normalize_simulator_rewards=False, buffer_size=500, random_start_simulator=False,
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
                                           use_residual_connection=True, training_epochs=50001,
                                           num_data_steps=4001, get_best_policy_flag=False, model_name=self.model_name,
                                           max_episode_len=self.args.max_steps, optimizing_specification=optimization_specification)
        fsc, extraction_stats = direct_extractor.clone_and_generate_fsc_from_policy(
            policy, environment, tf_environment)
        extraction_stats.store_as_json(self.model_name, "experiments_loopy_fscs")
        return fsc, extraction_stats
    
    def train_agent(self, agents_wrapper : AgentsWrapper, nr_rl_iterations : int = 1000, storm_control : StormPOMDPControl = None, fsc = None):
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
    
    def compute_assignment_through_family(self, fsc_like, latent_dim=2, agents_wrapper : AgentsWrapper = None, paynt_timeout=60):
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
    
    def compute_number_of_values_in_paynt_export(self, policy):
        nr_values = 0
        for obs in range(len(policy)):
            for mem in range(len(policy[obs])):
                nr_values += len(policy[obs][mem].keys())
        return nr_values

    
    def create_paynt_export(self, result, dtmc_state_to_state_and_mem):
        values = result.get_values()
        max_mem = np.max(dtmc_state_to_state_and_mem[:, 1]) + 1
        # for obs in range(self.quotient.pomdp.nr_observations):
        #     mem_info = [ {} for _ in range(max_mem) ]
        #     policy.append(mem_info)
        policy = [
            [{} for _ in range(max_mem)]
            for _ in range(self.quotient.pomdp.nr_observations)
        ]
        with open(f"policy_shape_{self.model_name}.txt", "w") as f:
            f.write(str(policy))
        import tqdm
        # tqdm.tqdm.write(f"Creating paynt export for {self.quotient.pomdp.nr_observations} observations and {max_mem} memory nodes.")
        continue_progress = tqdm.tqdm(total=len(values), desc="Creating paynt export")
        continue_progress.set_description(f"Creating paynt export for {self.quotient.pomdp.nr_observations} observations and {max_mem} memory nodes.")
        continue_progress.refresh()
        
        for dtmc_state in range(len(values)):
            continue_progress.update(1)
            value = values[dtmc_state]
            mdp_state = dtmc_state_to_state_and_mem[dtmc_state, 0]
            # mdp_choice = dtmc.quotient_choice_map[dtmc_state]
            memory_node = dtmc_state_to_state_and_mem[dtmc_state, 1]

            # memory_node = self.quotient.pomdp_manager.state_memory[mdp_state]
            observation = self.quotient.pomdp.observations[mdp_state]
            policy[observation][memory_node][mdp_state] = value

        return policy
    
    def create_paynt_export_vec(self, result, dtmc_state_to_state_and_mem):
        start_time = time.time()
        values = np.array(result.get_values())
        max_mem = np.max(dtmc_state_to_state_and_mem[:, 1]) + 1
        observations = np.array(self.quotient.pomdp.observations)

        policy = [
            [{} for _ in range(max_mem)]
            for _ in range(self.quotient.pomdp.nr_observations)
        ]
        dtmc_states = np.arange(len(values))
        mdp_states = dtmc_state_to_state_and_mem[dtmc_states, 0]
        memory_nodes = dtmc_state_to_state_and_mem[dtmc_states, 1]
        observations_for_states = observations[mdp_states]
        pairs = np.stack((observations_for_states, memory_nodes), axis=1)
        unique_pairs = np.unique(pairs, axis=0)
        for pair in unique_pairs:
            observation, memory_node = pair
            matching_indices = np.where((observations_for_states == observation) & (memory_nodes == memory_node))
            values_for_pair = values[matching_indices]
            real_mdp_states = mdp_states[matching_indices]
            policy[observation][memory_node] = dict(zip(real_mdp_states, values_for_pair))
        logger.info(f"Paynt export created in {time.time() - start_time:.2f} seconds.")
        return policy
        
    def compare_two_exports(self, export1, export2):
        if len(export1) != len(export2):
            logger.error(f"Export lengths do not match: {len(export1)} != {len(export2)}")
            return False
        for obs in range(len(export1)):
            if len(export1[obs]) != len(export2[obs]):
                logger.error(f"Export lengths for observation {obs} do not match: {len(export1[obs])} != {len(export2[obs])}")
                return False
            for mem in range(len(export1[obs])):
                if export1[obs][mem] != export2[obs][mem]:
                    logger.error(f"Export values for observation {obs} and memory {mem} do not match: {export1[obs][mem]} != {export2[obs][mem]}")
                    return False
        return True

    def compute_paynt_assignment_directly(self, fsc_like : TableBasedPolicy):
        
        fsc = ConstructorFSC.construct_fsc_from_table_based_policy(
            fsc_like, self.quotient)
        logger.info(f"Extracted FSC from RL policy.")
        logger.info(f"DTMC extraction")
        dtmc, dtmc_state_to_state_and_mem = self.quotient.get_induced_dtmc_from_fsc_vec(fsc)
        logger.info(f"Extracted DTMC: {dtmc}")
        logger.info(f"Checking DTMC.")
        result = stormpy.model_checking(dtmc, self.quotient.specification.optimality.formula)
        logger.info(f"DTMC check finished")
        logger.info(f"Result: {result.at(0)}")

        paynt_export_vec = self.create_paynt_export_vec(result, dtmc_state_to_state_and_mem)
        # paynt_export = self.create_paynt_export_vec(result, dtmc_state_to_state_and_mem)
        logger.info(f"Paynt export created.")
        return paynt_export_vec, result.at(0), fsc


    def compute_paynt_assignment_from_fsc_like(self, fsc_like : TableBasedPolicy, latent_dim=2, agents_wrapper : AgentsWrapper = None, paynt_timeout=60, old=False):
        if old:
            return self.compute_assignment_through_family(fsc_like, latent_dim, agents_wrapper, paynt_timeout), 0
        else:
            logger.info(f"Computing assignment from fsc_like.")
            return self.compute_paynt_assignment_directly(fsc_like)
        
    def single_shot_synthesis(self, agents_wrapper: AgentsWrapper, nr_rl_iterations: int, paynt_timeout: int, fsc=None, storm_control=None,
                              bottlenecking = False, self_interpretation = False):
        self.use_one_hot_memory = False
        self.train_agent(agents_wrapper, nr_rl_iterations, storm_control, fsc)

        agents_wrapper.agent.set_agent_greedy()
        latent_dim = 2 if not self.use_one_hot_memory else 3 ** 2

        agents_wrapper.agent.set_agent_greedy()
        agents_wrapper.agent.set_policy_masking()
        if bottlenecking:
            bottleneck_extractor, extracted_fsc, _, latent_dim = self.perform_bottleneck_extraction(
                agents_wrapper)
        elif self_interpretation:
            extracted_fsc, _ = self.perform_rl_to_fsc_cloning(
                agents_wrapper.agent.wrapper, 
                agents_wrapper.agent.environment, 
                agents_wrapper.agent.tf_environment, 
                latent_dim=latent_dim)
        else:
            aalpy_extractor = AALpyExtractor(
                agents_wrapper.agent.environment, agents_wrapper.agent.args, num_envs=agents_wrapper.agent.environment.num_envs)
            
            extracted_fsc = aalpy_extractor.extract_fsc(
                agents_wrapper.agent.wrapper)


        k_fsc = 3 ** latent_dim if not self.use_one_hot_memory else latent_dim
        logger.info(f"Extracted {k_fsc}-FSC from RL policy.")
        agents_wrapper.agent.set_agent_stochastic()
        paynt_export, value, classic_fsc = self.compute_paynt_assignment_from_fsc_like(extracted_fsc, latent_dim=latent_dim, agents_wrapper=agents_wrapper)
        logger.info(f"Paynt assignment computed.")
        return paynt_export, value, classic_fsc

    def run(self, multiple_assignments_benchmark = False, bottlenecking = False):
        if not hasattr(self.quotient, "pomdp"):
            pomdp = self.quotient.quotient_mdp
        else:
            pomdp = self.quotient.pomdp
        agents_wrapper = AgentsWrapper(
            pomdp, self.args, agent_folder=self.model_name)
        self.set_agents_wrapper(agents_wrapper)
        start_time = time.time()
        fsc = None
        while True:
            paynt_export, value, classic_fsc = self.single_shot_synthesis(
                agents_wrapper, 100, self.fsc_synthesis_time_limit, fsc,
                bottlenecking=bottlenecking, self_interpretation=True)
            print("RL Value:", value)
            if paynt_export is not None:
                agents_wrapper.agent.evaluation_result.add_paynt_bound(value)
                # print(fsc.observation_labels)
                # print(fsc.action_labels)

            if not self.loop:
                break
            if time.time() - start_time > self.time_limit:
                break
        # Save the final json file
        agents_wrapper.save_to_json(
            self.args.agent_name, model=self.model_name, method=self.args.learning_method)
        return paynt_export, value, classic_fsc

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
