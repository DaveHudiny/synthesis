# using Paynt for POMDP sketches

from compact_rl.robust_rl.robust_rl_tools import create_json_file_name, assignment_to_pomdp, generate_table_based_fsc_from_paynt_fsc
import paynt.quotient.fsc
import paynt.synthesizer.synthesizer_ar

from compact_rl.robust_rl.rnn_analyzer import RNNAnalyzer

import os


from compact_rl.rl.environment.environment_wrapper_vec import EnvironmentWrapperVec
from compact_rl.rl.environment.tf_py_environment import TFPyEnvironment

from paynt.rl_extension.self_interpretable_interface.black_box_extraction import BlackBoxExtractor

from paynt.rl_extension.family_extractors.direct_fsc_construction import ConstructorFSC

from paynt.rl_extension.robust_rl.family_quotient_numpy import FamilyQuotientNumpy
from paynt.quotient.pomdp_family import PomdpFamilyQuotient


from compact_rl.rl.agents.recurrent_ppo_agent import Recurrent_PPO_agent
from compact_rl.rl.tools.args_emulator import ArgsEmulator

import numpy as np

import logging


from compact_rl.robust_rl.benchmark_stats import BenchmarkStats

from paynt.quotient.fsc import FscFactored

from compact_rl.robust_rl.config_file import Config

from compact_rl.rl.interpreters.extracted_fsc.table_based_policy import TableBasedPolicy
from compact_rl.rl.tools.evaluators import evaluate_policy_in_model

from tf_agents.policies import TFPolicy


logger = logging.getLogger(__name__)


class RobustTrainer:
    def __init__(self, args: ArgsEmulator, use_one_hot_memory=False, latent_dim=2,
                 pomdp_sketch=None,
                 obs_evaluator=None, quotient_state_valuations=None,
                 family_quotient_numpy: FamilyQuotientNumpy = None,
                 use_gumbel_softmax=False, use_vq_vae=False):
        self.args = args
        self.use_one_hot_memory = use_one_hot_memory
        self.model_name = args.model_name
        self.pomdp_sketch = pomdp_sketch
        self.obs_evaluator = obs_evaluator
        self.quotient_state_valuations = quotient_state_valuations
        self.family_quotient_numpy = family_quotient_numpy
        fsc_size = latent_dim if use_one_hot_memory else 3**latent_dim
        self.cut_probs = 0.0

        if self.args.extraction_type == "alergia":
            self.autlearn_extraction = True
        else:
            self.autlearn_extraction = False
        self.extraction_type = args.extraction_type
        self.use_gumbel_softmax = use_gumbel_softmax
        self.use_vq_vae = use_vq_vae
        self.direct_extractor = self.init_extractor(
            latent_dim, self.autlearn_extraction)
        self.period_between_worst_case_evaluation = 5

        self.benchmark_stats = BenchmarkStats(
            fsc_size=fsc_size, num_training_steps_per_iteration=301,
            batched_vec_storm=args.batched_vec_storm, extraction_type=args.extraction_type,
            lstm_width=args.width_of_lstm, without_extraction=args.without_extraction,
            geometric_batched_vec_storm=args.geometric_batched_vec_storm,
            periodic_restarts=args.periodic_restarts, period_between_worst_case_evaluation=self.period_between_worst_case_evaluation,
            seed=args.seed)
        self.agent = None
        self.extraction_less = args.without_extraction

    def save_stats(self, path):
        self.benchmark_stats.save_stats(path)

    def init_extractor(self, latent_dim, autlearn_extraction=False):
        if not self.args.extraction_type == "bottleneck":
            direct_extractor = BlackBoxExtractor(memory_len=latent_dim, is_one_hot=self.use_one_hot_memory,
                                                          use_residual_connection=True, training_epochs=20001,
                                                          num_data_steps=self.args.max_steps * 4, get_best_policy_flag=False, model_name=self.model_name,
                                                          max_episode_len=self.args.max_steps,
                                                          family_quotient_numpy=self.family_quotient_numpy,
                                                          autlearn_extraction=autlearn_extraction,
                                                          use_gumbel_softmax=self.use_gumbel_softmax,
                                                          use_vq_vae=self.use_vq_vae)
            return direct_extractor
        else:
            return None

    def eval_policy_wrapper(self, policy : TFPolicy, environment : EnvironmentWrapperVec, batch_size=512):
        """
        Evaluates a policy in the environment and returns the average return and reachability.
        """
        environment.temporarily_set_num_envs(batch_size)
        tf_environment = TFPyEnvironment(environment)
        evaluation_result = evaluate_policy_in_model(
            policy, self.args, environment, tf_environment, max_steps=self.args.max_steps)
        environment.reset_num_envs()
        return evaluation_result.returns[-1], evaluation_result.reach_probs[-1]

    def call_si_or_aalpy(self, agent: Recurrent_PPO_agent, environment: EnvironmentWrapperVec, tf_environment: TFPyEnvironment, num_data_steps=4001, training_epochs=10001):
        self.direct_extractor.num_data_steps = num_data_steps
        self.direct_extractor.training_epochs = training_epochs
        policy = agent.get_policy(False, True)
        fsc, extraction_stats = self.direct_extractor.clone_and_generate_fsc_from_policy(
            policy, environment, tf_environment)
        self.benchmark_stats.add_extracted_fsc_performance(
            extraction_stats.extracted_fsc_reward[-1])
        self.benchmark_stats.add_extracted_fsc_reachability(
            extraction_stats.extracted_fsc_reachability[-1])
        if len(extraction_stats.lstm_extracted_reachability) > 0 and len(extraction_stats.lstm_extracted_return) > 0:
            self.benchmark_stats.add_lstm_extracted_results(
                extraction_stats.lstm_extracted_reachability[-1], extraction_stats.lstm_extracted_return[-1])
        if extraction_stats.large_fsc_extracted_reachabilities and extraction_stats.large_fsc_extracted_returns:
            for memory_size in extraction_stats.large_fsc_extracted_reachabilities:
                self.benchmark_stats.add_large_fsc_extracted_results(
                    memory_size, extraction_stats.large_fsc_extracted_reachabilities[memory_size][-1], extraction_stats.large_fsc_extracted_returns[memory_size][-1])

        
        return fsc

    def extract_fsc(self, agent: Recurrent_PPO_agent, environment: EnvironmentWrapperVec, quotient, 
                    num_data_steps=4001, training_epochs=10001, get_dict=False,
                    use_masking: bool = True) -> paynt.quotient.fsc.FscFactored:
        if not self.extraction_type == "bottleneck":
            self.direct_extractor.num_data_steps = num_data_steps
            self.direct_extractor.training_epochs = training_epochs
        # agent.set_agent_greedy()
        # agent.set_policy_masking()
        if use_masking:
            agent.set_policy_masking()
        else:
            agent.unset_policy_masking()

        tf_environment = TFPyEnvironment(environment)
        print("Calling call_si_or_aalpy")
        fsc = self.call_si_or_aalpy(
            agent, environment, tf_environment, num_data_steps=num_data_steps, training_epochs=training_epochs)
        print("Finished call_si_or_aalpy, constructing PAYNT representation...")
        paynt_fsc = ConstructorFSC.construct_fsc_from_table_based_policy(
            fsc, quotient, family_quotient_numpy=self.family_quotient_numpy, cut_probs=self.cut_probs)
        print("Finished constructing PAYNT representation. Computing available nodes in FSC...")
        
        available_nodes = paynt_fsc.compute_available_updates(0)
        self.benchmark_stats.available_nodes_in_fsc.append(available_nodes)
        if get_dict:
            return {
                "extracted_paynt_fsc": paynt_fsc,
                "extracted_fsc": fsc
            }
        print(f"Available nodes in FSC: {available_nodes}")
        return paynt_fsc

    def train_on_new_pomdp(self, pomdp=None, agent: Recurrent_PPO_agent = None, nr_iterations=1500):
        # environment = EnvironmentWrapperVec(pomdp, self.args, num_envs=256, enforce_compilation=True,
        #                                     obs_evaluator=self.obs_evaluator,
        #                                     quotient_state_valuations=self.quotient_state_valuations,
        #                                     observation_to_actions=self.pomdp_sketch.observation_to_actions)
        if pomdp is not None and self.args.batched_vec_storm:
            self.environment.add_new_pomdp(pomdp)
        elif pomdp is not None and not self.args.batched_vec_storm:
            self.environment = EnvironmentWrapperVec(pomdp, self.args, num_envs=self.args.num_environments, enforce_compilation=True,
                                                     obs_evaluator=self.obs_evaluator,
                                                     quotient_state_valuations=self.quotient_state_valuations,
                                                     observation_to_actions=self.pomdp_sketch.observation_to_actions)
            agent.change_environment(self.environment)
        else:
            logger.info("No POMDP provided, using existing environment.")
        agent.train_agent(iterations=nr_iterations)
        self.benchmark_stats.add_rl_performance(
            np.abs(agent.evaluation_result.returns[-1]))
        self.benchmark_stats.add_rl_performance_reachability(
            np.abs(agent.evaluation_result.reach_probs[-1]))

    def generate_agent(self, pomdp, args: ArgsEmulator) -> Recurrent_PPO_agent:
        self.environment = EnvironmentWrapperVec(pomdp, args, num_envs=args.num_environments, enforce_compilation=True,
                                                 obs_evaluator=self.obs_evaluator,
                                                 quotient_state_valuations=self.quotient_state_valuations,
                                                 observation_to_actions=self.pomdp_sketch.observation_to_actions)
        self.tf_env = TFPyEnvironment(self.environment)
        self.agent = Recurrent_PPO_agent(
            environment=self.environment, tf_environment=self.tf_env, args=args)
        return self.agent

    def add_new_pomdp(self, pomdp, agent: Recurrent_PPO_agent):
        """
        Adds a new POMDP to the environment.
        """
        if self.args.batched_vec_storm:
            agent.environment.add_new_pomdp(pomdp)
        else:
            agent.environment = EnvironmentWrapperVec(pomdp, self.args, num_envs=256, enforce_compilation=True,
                                                      obs_evaluator=self.obs_evaluator,
                                                      quotient_state_valuations=self.quotient_state_valuations,
                                                      observation_to_actions=self.pomdp_sketch.observation_to_actions)
            agent.change_environment(self.environment)

    def add_initial_pomdps(self, pomdp, pomdp_sketch: PomdpFamilyQuotient, nr_initial_pomdps=10):
        for _ in range(nr_initial_pomdps):
            hole_assignment = pomdp_sketch.family.pick_random()
            pomdp, _, _ = assignment_to_pomdp(
                pomdp_sketch, hole_assignment)
            self.add_new_pomdp(pomdp, self.agent)

    def extraction_loop(self, pomdp_sketch, project_path, nr_initial_pomdps=10, num_samples_learn=401):
        """
        Main extraction loop for robust RL.
        """
        logger.info("Starting extraction loop")
        rnn_analyzer = RNNAnalyzer(self.args)
        args_emulated = self.args
        if project_path.split("/")[-1] == "":
            config = Config(project_path.split("/")[-2])
        else:
            config = Config(project_path.split("/")[-1])
        json_path = create_json_file_name(
            f"{project_path}", seed=f"{self.args.seed}")

        hole_assignment = pomdp_sketch.family.pick_random()

        pomdp, _, _ = assignment_to_pomdp(pomdp_sketch, hole_assignment)
        if nr_initial_pomdps:  # Add nr_initial_pomdps random POMDPs to the environment
            self.add_initial_pomdps(pomdp, pomdp_sketch, nr_initial_pomdps)
        nr_iterations = config.nr_initial_iter
        # Upper limit of the outer iterations. In practice, we are stopped by an external timeout.
        for i in range(101):
            self.clean_cache()

            logger.info(f"Iteration {i+1} of extraction RL loop")

            if args_emulated.single_pomdp_experiment:
                pomdp = None
            print("Training on new POMDP...")
            self.train_on_new_pomdp(  # Train the agent on multiple POMDPs
                pomdp, self.agent, nr_iterations=nr_iterations)

            # Analysis of the dormant neurons. Not mentioned in the paper, but used during the tuning.
            # nr_clusters = rnn_analyzer.analyze(self.agent, self.tf_env)
            # self.benchmark_stats.add_nr_clusters(nr_clusters)

            nr_iterations = config.nr_inner_iter if not self.args.periodic_restarts else 401

            # In our final implementation, we always use masking for the extraction
            for use_masking in [True]:
                print("Extracting FSC...")
                fsc = self.extract_fsc(self.agent, self.agent.environment, pomdp_sketch, get_dict=True,
                                       num_data_steps=num_samples_learn, use_masking=use_masking, training_epochs=config.extraction_epochs)
                # Evaluate the FSC on all POMDPs
                paynt_fsc = fsc["extracted_paynt_fsc"]
                table_based_fsc = fsc["extracted_fsc"]
                print("Building DTMC sketch and synthesizing assignment...")
                dtmc_sketch = pomdp_sketch.build_dtmc_sketch(
                    paynt_fsc, negate_specification=True)
                print("DTMC sketch built, starting synthesis...")
                synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(
                    dtmc_sketch)
                hole_assignment = synthesizer.synthesize(keep_optimum=True)

            logger.info(
                f"Extracted FSC for hole assignment: {hole_assignment}")
            self.benchmark_stats.add_family_performance(
                synthesizer.best_assignment_value)

            if not self.extraction_less:
                pomdp, _, _ = assignment_to_pomdp(
                    pomdp_sketch, hole_assignment)
            else:
                hole_assignment = pomdp_sketch.family.pick_random()
                pomdp, _, _ = assignment_to_pomdp(
                    pomdp_sketch, hole_assignment)

            self.benchmark_stats.shrink_and_perturb_activated.append(False)

            self.agent.evaluation_result.save_to_json(
                json_path, new_pomdp=True)

            self.save_stats(json_path)
            if self.args.periodic_restarts:
                self.agent.reset_weights()
            del synthesizer
            del dtmc_sketch

    def train_and_extract_single_pomdp(self, pomdp_sketch: PomdpFamilyQuotient, nr_iterations=1500, num_samples_learn=4001, args: ArgsEmulator = None, project_path: str = None):
        """
        RL training and extraction on a single POMDP. Loopless.
        """
        rnn_analyzer = RNNAnalyzer(self.args)

        self.train_on_new_pomdp(None, self.agent, nr_iterations=nr_iterations)
        rnn_analyzer.analyze(self.agent, self.tf_env)
        fsc = self.extract_fsc(self.agent, self.agent.environment, pomdp_sketch,
                               num_data_steps=num_samples_learn, training_epochs=6001, get_dict=True)
        paynt_fsc = fsc["extracted_paynt_fsc"]

        dtmc_sketch = pomdp_sketch.build_dtmc_sketch(
            paynt_fsc)
        one_by_one = paynt.synthesizer.synthesizer_onebyone.SynthesizerOneByOne(
            dtmc_sketch)
        hole_assignment = one_by_one.synthesize(keep_optimum=True)
        self.benchmark_stats.add_family_performance(
            one_by_one.best_assignment_value)
        logger.info(
            f"Synthesized assignment: {hole_assignment} with value {one_by_one.best_assignment_value}")
        json_path = create_json_file_name(
            f"{project_path}", seed=f"{self.args.seed}")
        self.agent.evaluation_result.save_to_json(json_path, new_pomdp=False)
        self.save_stats(json_path)
        return paynt_fsc, hole_assignment, one_by_one.best_assignment_value

    def clean_cache(self):
        """
        Cleans the cache in TensorFlow keras and JAX.
        """
        import tensorflow as tf
        import jax
        jax.clear_backends()
        tf.keras.backend.clear_session()


def initialize_extractor(pomdp_sketch, args_emulated: ArgsEmulator, family_quotient_numpy: FamilyQuotientNumpy):
    if family_quotient_numpy is not None:
        quotient_sv = pomdp_sketch.quotient_mdp.state_valuations
        quotient_obs = pomdp_sketch.obs_evaluator
    else:
        quotient_sv = None
        quotient_obs = None

    use_one_hot_memory = True if args_emulated.extraction_type in ("si-g", "vq-vae") else False
    use_gumbel_softmax = True if args_emulated.extraction_type == "si-g" else False
    use_vq_vae = True if args_emulated.extraction_type == "vq-vae" else False


    if "avoid-large" in args_emulated.prism_model or "drone-2-6-1" in args_emulated.prism_model or "moving-obstacles" in args_emulated.prism_model:
        latent_dim = 10  # For these models, we use larger latent dimension => usually larger FSCs
    else:
        latent_dim = 3

    extractor = RobustTrainer(args_emulated, use_one_hot_memory=use_one_hot_memory, latent_dim=latent_dim, quotient_state_valuations=quotient_sv,
                              obs_evaluator=quotient_obs, pomdp_sketch=pomdp_sketch,
                              family_quotient_numpy=family_quotient_numpy, use_gumbel_softmax=use_gumbel_softmax,
                              use_vq_vae=use_vq_vae)

    return extractor

