from collections import defaultdict
from enum import Enum
from typing import Union, Dict, List

import numpy as np

import tensorflow as tf
from aalpy import MealyMachine, Onfsm, StochasticMealyMachine, StochasticMealyState, run_RPNI, run_Alergia
from aalpy.learning_algs import run_k_tails
from aalpy.learning_algs.general_passive.GeneralizedStateMerging import GeneralizedStateMerging
from aalpy.learning_algs.general_passive.GsmNode import GsmNode
from aalpy.learning_algs.general_passive.ScoreFunctionsGSM import differential_info, hoeffding_compatibility, \
    ScoreCalculation, ScoreFunction
from scipy.stats import chi2

from comp_checker import ChiSquareChecker
from compact_rl.rl.interpreters.direct_fsc_extraction.test_functions import *
from compact_rl.rl.interpreters.direct_fsc_extraction.encoding_functions import get_encoding_functions

from tf_agents.policies import TFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy

from compact_rl.rl.environment.environment_wrapper_vec import EnvironmentWrapperVec
from learn_aut import create_trajectories, create_mealy_learn_traj
from compact_rl.rl.tools.evaluation_results_class import EvaluationResults

from compact_rl.rl.tests.general_test_tools import init_environment, init_args
from compact_rl.rl.agents.recurrent_ppo_agent import Recurrent_PPO_agent
from compact_rl.rl.tools.evaluators import evaluate_policy_in_model

from compact_rl.rl.interpreters.bottlenecking.quantized_bottleneck_extractor import TableBasedPolicy
from compact_rl.rl.tools.specification_check import SpecificationChecker

import logging

import os

from compact_rl.rl.interpreters.direct_fsc_extraction.extraction_stats import ExtractionStats
from compact_rl.rl.interpreters.direct_fsc_extraction.data_sampler import sample_data_with_policy
from compact_rl.rl.interpreters.direct_fsc_extraction.cloned_fsc_actor_policy import ClonedFSCActorPolicy
from compact_rl.rl.interpreters.direct_fsc_extraction.cloned_gru_network_policy import ClonedGRUActorPolicy

from compact_rl.rl.interpreters.aalpy_extraction.mealy_automata_learner import MealyAutomataLearner

from compact_rl.rl.agents.policies.policy_mask_wrapper import PolicyMaskWrapper

from paynt.rl_extension.robust_rl.family_quotient_numpy import FamilyQuotientNumpy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBUG = False

class AutLearn(Enum):
    KTAIL = 1
    SMM = 2
    MDP = 3
    DET = 4

class BlackBoxExtractor:

    def __init__(self, memory_len = 1, is_one_hot = False, use_residual_connection = False,
                 training_epochs = 100000, num_data_steps = 6000,
                 get_best_policy_flag = False, model_name = "generic_model",
                 max_episode_len = 800, 
                 optimizing_specification : SpecificationChecker.Constants = SpecificationChecker.Constants.REACHABILITY,
                 family_quotient_numpy : FamilyQuotientNumpy = None, autlearn_extraction = True,
                 use_gumbel_softmax = False, use_vq_vae = False, stacked_observations = False, seed=42, non_deterministic=True):
        self.autlearn_extraction = autlearn_extraction
        self.iteration = 0
        self.memory_len = memory_len
        self.is_one_hot = is_one_hot
        self.use_residual_connection = use_residual_connection
        self.regenerate_fsc_network_flag = False # This flag ensures, that if we repeatedly clone the same policy
                                                # we do not regenerate new FSC network every time.
        self.cloned_actor : ClonedFSCActorPolicy = None
        self.specification_checker : SpecificationChecker = None
        self.training_epochs = training_epochs
        self.num_data_steps = num_data_steps
        self.get_best_policy_flag = get_best_policy_flag
        self.model_name = model_name
        self.max_episode_len = max_episode_len
        self.extraction_stats = None
        self.optimizing_specification = optimizing_specification
        self.family_quotient_numpy = family_quotient_numpy
        self.aut_learn_type = AutLearn.SMM
        self.joint_action_update = False
        self.use_gumbel_softmax = use_gumbel_softmax
        self.use_vq_vae = use_vq_vae
        self.stacked_observations = stacked_observations
        self.non_deterministic = non_deterministic
        # VQ-VAE, like the Gumbel-Softmax bottleneck, uses one-hot memory nodes, but its
        # nearest-code assignment is always hard, so memory updates are deterministic
        # (complete_probs=False) rather than stochastic distributions over N.
        self.complete_probs = True if use_gumbel_softmax else False
        self.seed = seed

    def set_memory_len(self, memory_len):
        self.memory_len = memory_len
        self.regenerate_fsc_network_flag = True

    def set_is_one_hot(self, is_one_hot):
        self.is_one_hot = is_one_hot
        self.regenerate_fsc_network_flag = True

    def set_use_residual_connection(self, use_residual_connection):
        self.use_residual_connection = use_residual_connection
        self.regenerate_fsc_network_flag = True

    def force_new_fsc(self):
        self.regenerate_fsc_network_flag = True

    def set_specification_checker(self, specification_checker : SpecificationChecker):
        self.specification_checker = specification_checker

    def reset_cloned_actor(self, original_policy : TFPolicy, 
                           orig_eval_result : EvaluationResults, 
                           env : EnvironmentWrapperVec):
        self.cloned_actor = ClonedFSCActorPolicy(
                original_policy, self.memory_len, original_policy.observation_and_action_constraint_splitter,
                use_one_hot=self.is_one_hot, use_residual_connection=self.use_residual_connection,
                optimization_specification = self.optimizing_specification, model_name = self.model_name,
                find_best_policy=self.get_best_policy_flag,
                max_episode_length=self.max_episode_len, observation_length=env.observation_spec_len, 
                orig_env_use_stacked_observations=self.stacked_observations,
                use_gumbel_softmax=self.use_gumbel_softmax, use_vq_vae=self.use_vq_vae, seed=self.seed)
        self.extraction_stats = ExtractionStats(original_policy_reachability=orig_eval_result.reach_probs[-1],
                                                     original_policy_reward=orig_eval_result.returns[-1],
                                                     use_one_hot=self.is_one_hot,
                                                     number_of_samples=self.num_data_steps * env.num_envs,
                                                     memory_size=self.memory_len,
                                                     residual_connection=self.use_residual_connection)
        self.regenerate_fsc_network_flag = False


    def learn_fsc(self, all_trajectories : List, original_policy : TFPolicy,
                                           env : EnvironmentWrapperVec):
        
        print(f"Learning from {len(all_trajectories)} trajectories")
        if self.aut_learn_type == AutLearn.KTAIL:

            model = run_k_tails(all_trajectories, 'mealy', k=1,
                                input_completeness=None, print_info=True)
        elif self.aut_learn_type == AutLearn.SMM:

            model = run_Alergia(all_trajectories,"smm", compatibility_checker = ChiSquareChecker(alpha=0.01))
            print(f"Learned Chi2 SMM of size {len(model.states)}")
            epsilon = 0.5
            score = ScoreCalculation(hoeffding_compatibility(epsilon, True))
            def likelihood_ratio_score(alpha=0.05) -> ScoreFunction:
                if not 0 < alpha <= 1:
                    raise ValueError(f"Confidence {alpha} not between 0 and 1")

                def score_fun(part: Dict[GsmNode, GsmNode]):
                    llh_diff, param_diff = differential_info(part)
                    if param_diff == 0:
                        # This should cover the corner case when the partition merges only states with no outgoing transitions.
                        return -1  # Let them be very bad merges.
                    score = 1 - chi2.cdf(2 * llh_diff, param_diff)
                    if score < alpha:
                        return False
                    return score

                return score_fun
        elif self.aut_learn_type == AutLearn.MDP:
            mdp_traj = [["init"] + t for t in all_trajectories]
            model = run_Alergia(mdp_traj, "mdp", compatibility_checker=ChiSquareChecker(alpha=0.01))
            print(f"Learned MDP of size {len(model.states)}")

        elif self.aut_learn_type == AutLearn.DET:
            learn_trajectories = create_mealy_learn_traj(all_trajectories)
            model = run_RPNI(learn_trajectories, 'mealy', algorithm='gsm',
                             input_completeness=None, print_info=True)
        self.make_aalpy_input_complete(model,env.stormpy_model.nr_observations)
        if self.aut_learn_type == AutLearn.KTAIL:
            probs = self.compute_probs(model,all_trajectories)
        else:
            probs = None
        print(f"Learned FSC of size {len(model.states)}")
        self.iteration += 1
        fsc_actions, fsc_updates, initial_state = self.aalpy_to_fsc(model,env,probs)
        update_shape = None if fsc_updates is None else fsc_updates.shape
        print(f"FSC actions shape: {fsc_actions.shape}, updates shape: {update_shape}, initial state: {initial_state}")
        nr_observations = len(self.family_quotient_numpy.observation_to_legal_action_mask) if self.family_quotient_numpy is not None else env.stormpy_model.nr_observations
        table_based_policy = TableBasedPolicy(
            original_policy, fsc_actions, fsc_updates, initial_memory=initial_state, action_keywords=env.action_keywords,
            nr_observations=nr_observations)
        # print(f"New FSC actions shape: {table_based_policy.tf_observation_to_action_table.shape}, updates shape: {table_based_policy.tf_observation_to_update_table.shape}, initial state: {table_based_policy.initial_memory}")

        return table_based_policy,model

    def aalpy_to_fsc(self, model : Union[MealyMachine,Onfsm,StochasticMealyMachine],
                     env : EnvironmentWrapperVec,probs):
        n_states = len(model.states)
        s_id_to_fsc_id = {state.state_id : fsc_id for fsc_id, state in enumerate(model.states)}
        nr_observations = env.stormpy_model.nr_observations
        if self.joint_action_update:
            fsc_actions = np.zeros((n_states, nr_observations, n_states * len(env.action_keywords)), dtype=np.float32)
            fsc_updates = None
        else:
            fsc_actions = np.zeros((n_states, nr_observations, len(env.action_keywords)), dtype=np.float32)
            fsc_updates = np.zeros((n_states, nr_observations, n_states), dtype=np.float32)

        for state in model.states:
            fsc_id = s_id_to_fsc_id[state.state_id]
            for obs in range(nr_observations):
                str_obs = str(obs)
                if self.aut_learn_type == AutLearn.KTAIL:
                    for (act, next_state) in state.transitions[str_obs]:
                        action_int = int(act) if act != "epsilon" else 0
                        next_fsc_id = s_id_to_fsc_id[next_state.state_id]
                        prob = probs[(state.state_id, str_obs, act)]
                        fsc_actions[fsc_id, obs, action_int] = prob
                        fsc_updates[fsc_id, obs, next_fsc_id] = prob
                elif self.aut_learn_type == AutLearn.SMM:
                    for (next_state,act,prob) in state.transitions[str_obs]:
                        action_int = int(act) if act != "epsilon" else 0
                        next_fsc_id = s_id_to_fsc_id[next_state.state_id]
                        if self.joint_action_update:
                            fsc_actions[fsc_id, obs, action_int * n_states + next_fsc_id] = prob
                        else:
                            fsc_actions[fsc_id, obs, action_int] = prob
                            fsc_updates[fsc_id, obs, next_fsc_id] = prob
                elif self.aut_learn_type == AutLearn.MDP:
                    for (next_state,prob) in state.transitions[str_obs]:
                        act = next_state.output
                        # act can be == "Init" if there is input completion, i.e., observation completion for some obs
                        action_int = int(act) if act != "epsilon" and act != "init" else 0
                        next_fsc_id = s_id_to_fsc_id[next_state.state_id]
                        if self.joint_action_update:
                            fsc_actions[fsc_id, obs, action_int * n_states + next_fsc_id] = prob
                        else:
                            fsc_actions[fsc_id, obs, action_int] = prob
                            fsc_updates[fsc_id, obs, next_fsc_id] = prob
                elif self.aut_learn_type == AutLearn.DET:
                    if self.joint_action_update:
                        raise Exception("Not Implemented")
                    next_state = state.transitions[str_obs]
                    action = int(state.output_fun[str_obs]) if state.output_fun[str_obs] != "epsilon" else 0
                    next_fsc_id = s_id_to_fsc_id[next_state.state_id]
                    fsc_actions[fsc_id, obs, action] = 1
                    fsc_updates[fsc_id, obs, next_fsc_id] = 1

        # add full action support
        fsc_actions += 0.0001
        if self.joint_action_update and (self.aut_learn_type == AutLearn.SMM or self.aut_learn_type == AutLearn.MDP):
            fsc_actions = fsc_actions / np.sum(fsc_actions, axis=-1, keepdims=True)
        else:
            for state in range(n_states):
                for obs in range(nr_observations):
                    fsc_actions[state,obs,:] /= sum(fsc_actions[state,obs,:])
        return fsc_actions, fsc_updates, s_id_to_fsc_id[model.initial_state.state_id]

    def make_aalpy_input_complete(self, model : Union[MealyMachine,Onfsm,StochasticMealyMachine], nr_observations):
        if self.aut_learn_type == AutLearn.SMM and len(model.states) == 1:
            print("Learned single state SMM")
            dummy_state = StochasticMealyState("q10")
            model.states.append(dummy_state)
            for obs in model.initial_state.transitions.keys():
                for (_,act,prob) in model.initial_state.transitions[obs]:
                    dummy_state.transitions[obs].append((dummy_state,act,prob))
                new_trans = [(dummy_state, act,prob) for(_,act,prob) in model.initial_state.transitions[obs]]
                model.initial_state.transitions[obs] = new_trans
        for state in model.states:
            for obs in range(nr_observations):
                obs = str(obs)
                if obs not in state.transitions.keys():
                    target_state = state
                    if self.aut_learn_type == AutLearn.KTAIL:
                        state.transitions[obs].append(( 'epsilon', target_state))
                    elif self.aut_learn_type == AutLearn.SMM:
                        state.transitions[obs].append((  target_state,'epsilon',1.0))
                    elif self.aut_learn_type == AutLearn.MDP:
                        state.transitions[obs].append((  target_state,1.0))
                    elif self.aut_learn_type == AutLearn.DET:
                        state.transitions[obs] = target_state
                        state.output_fun[obs] = 'epsilon'

    def compute_probs(self, model, learn_trajectories):
        counts = defaultdict(int)
        for trajectory in learn_trajectories:
            state = model.initial_state
            for (obs,act) in trajectory:
                transitions = state.transitions[obs]
                next_state = None
                for a,next_s in transitions:
                    if act == a:
                        counts[(state.state_id,obs,a)] += 1
                        next_state = next_s
                        break
                if next_state is None:
                    print("Did not find next state, how?")
                    exit(1)
                else:
                    state = next_state
        # normalize
        probs = dict()
        for state in model.states:
            counts_for_state = [((obs,act), cnt) for ((s_id,obs,act), cnt) in counts.items() if s_id == state.state_id]
            obs_set = {obs for ((obs,_),_) in counts_for_state}
            for obs in obs_set:
                cnt_sum = sum([cnt for ((o,a),cnt) in counts_for_state])
                for ((obs_i,act), cnt) in counts_for_state:
                    if obs_i == obs:
                        prob_for_a = cnt/cnt_sum
                        probs[(state.state_id,obs_i,act)] = prob_for_a
        return probs

    def create_alergia_smm_traj(self, all_trajectories):
        def create_single_traj(t):
            new_t = []
            for (i,o) in t:
                new_t.extend([i,o])
            return new_t
        smm_trajs = [create_single_traj(t) for t in all_trajectories]
        for t in smm_trajs:
            print(t)
        return smm_trajs
    
    def sample_subtrajectories(self, all_trajectories : list[tuple[str, int]], num_samples : int = 10000, fixed_length : int = 32):
        """
        Samples subtrajectories from the given trajectories.
        :param all_trajectories: List of trajectories to sample from.
        :param num_samples: Number of samples to take.
        :param fixed_length: Length of each sampled subtrajectory.
        :return: List of sampled subtrajectories.
        """
        sampled_sub_trajs = []
        num_trajectories = len(all_trajectories)
        for _ in range(num_samples):
            random_index = np.random.randint(0, num_trajectories)
            trajectory = all_trajectories[random_index]
            if len(trajectory) < fixed_length:
                sampled_sub_trajs.append(trajectory)
            else:
                start_index = np.random.randint(0, len(trajectory) - fixed_length + 1)
                sub_trajectory = [(trajectory[i][0], trajectory[i][1]) for i in range(start_index, start_index + fixed_length)]
                sampled_sub_trajs.append(sub_trajectory)
        logger.info(f"Sampled {len(sampled_sub_trajs)} sub-trajectories of length {fixed_length}")

        return sampled_sub_trajs
    
    def aalpy_extraction(self, original_policy : TFPolicy, 
                         env : EnvironmentWrapperVec, tf_env : TFPyEnvironment = None) -> tuple[TableBasedPolicy, ExtractionStats]:
        orig_eval_result = evaluate_policy_in_model(original_policy, environment=env, tf_environment=tf_env,
                                                        max_steps=(self.max_episode_len + 1) * 2)

        # if isinstance(original_policy, PolicyMaskWrapper):
        # original_policy.set_policy_masker()
        
        # original_policy.set_greedy(True)
        logger.info("Sampling data with original policy")
        use_replay_buffer = True
        sample_sub_trajs = False
        all_trajectories = []
        num_samples = self.num_data_steps
        for i in range(1):
                buffer = sample_data_with_policy(
                    original_policy, num_samples=num_samples, environment=env, tf_environment=tf_env,
                    use_replay_buffer=use_replay_buffer, get_both=False)
                if use_replay_buffer:
                    replay_buffer: TFUniformReplayBuffer = buffer
                    observations, actions = MealyAutomataLearner.convert_trajectories_to_episodes(replay_buffer.gather_all())
                    for obs_sequence, action_sequence in zip(observations, actions):
                        trajectory = list(zip(obs_sequence.tolist(), action_sequence.tolist()))
                        all_trajectories.append(trajectory)
                    print(f"Learned {len(all_trajectories)} trajectories")
                    print("Learned trajectory lengths ", set(map(len, all_trajectories)))
                    if sample_sub_trajs:
                        all_trajectories = self.sample_subtrajectories(all_trajectories, fixed_length=32, num_samples=20000)
                        print(f"Sampled {len(all_trajectories)} sub-trajectories of length 32")
                    del buffer
                else:
                    print(f"Buffer size: {len(buffer)}")
                    aut_learn_data = buffer
                    n_envs = env.num_envs
                    all_trajectories.extend(create_trajectories(aut_learn_data, mealy=True, n_envs=n_envs))
                    print(f"Learned {len(all_trajectories)} trajectories")
                    print("Learn trajectory lengths ",set(map(len,all_trajectories)))
                    del buffer
        print("All trajectories collected")
        # original_policy.set_greedy(False)
        logger.info("Data sampled")
        if isinstance(original_policy, PolicyMaskWrapper):
            original_policy.set_identity_masker()
        logger.info("Learning FSC from original policy")

        fsc, aalpy_model = self.learn_fsc(all_trajectories, original_policy, env)
            
        extraction_stats = ExtractionStats(
                original_policy_reachability=0,
                original_policy_reward=0,
                use_one_hot=False,
                number_of_samples=0,
                memory_size=len(aalpy_model.states),
                residual_connection=False
        )
        fsc_res = evaluate_policy_in_model(fsc, environment=env, tf_environment=tf_env,
                                               max_steps=(self.max_episode_len + 1) * 2)

        extraction_stats.add_fsc_result(fsc_res.reach_probs[-1], fsc_res.returns[-1])
        extraction_stats.add_number_of_training_trajectories(len(all_trajectories))
        del all_trajectories
        return fsc, extraction_stats
    
    def self_interpretable_extraction(self, original_policy : TFPolicy,
                                        env : EnvironmentWrapperVec, 
                                        tf_env : TFPyEnvironment = None, with_gru=False) -> tuple[TableBasedPolicy, ExtractionStats]:
        orig_eval_result = evaluate_policy_in_model(original_policy, environment=env, tf_environment=tf_env, max_steps=(self.max_episode_len + 1) * 2)
        if self.specification_checker is not None:
            self.specification_checker.set_optimal_value_from_evaluation_results(orig_eval_result)

        if self.regenerate_fsc_network_flag or self.cloned_actor is None:
            self.reset_cloned_actor(original_policy, orig_eval_result, env)

        logger.info("Sampling data with original policy")
        print(f"Sampling {self.num_data_steps} steps")
        buffer = sample_data_with_policy(
            original_policy, num_samples=self.num_data_steps, environment=env, tf_environment=tf_env)
        logger.info("Data sampled")
        logger.info("Cloning original policy to FSC")

        extraction_stats = self.cloned_actor.behavioral_clone_original_policy_to_fsc(
            buffer, num_epochs=self.training_epochs, specification_checker=self.specification_checker,
            environment=env, tf_environment=tf_env, args=None, extraction_stats=self.extraction_stats)
        
        
        if with_gru:
            cloned_lstm_actor = ClonedGRUActorPolicy(original_policy, observation_and_action_constraint_splitter=original_policy.observation_and_action_constraint_splitter,
                                                      observation_length=env.observation_spec_len, lstm_units=32)
            extraction_stats_lstm = cloned_lstm_actor.behavioral_clone_original_policy_to_fsc(
                buffer, num_epochs=5001, environment=env, tf_environment=tf_env, args=None, extraction_stats=None
            )
            extraction_stats.add_lstm_result(extraction_stats_lstm.extracted_policy_reachabilities[-1], extraction_stats_lstm.extracted_policy_rewards[-1])

        if False: # Experiment with larger FSCs with matrix setting
            for size in [9, 16, 25, 36, 49, 64]:
                fsc_actor_sized = ClonedFSCActorPolicy(
                    original_policy, size, original_policy.observation_and_action_constraint_splitter,
                    use_one_hot=self.is_one_hot, use_residual_connection=self.use_residual_connection,
                    optimization_specification = self.optimizing_specification, model_name = self.model_name + f"_aux_{size}",
                    find_best_policy=self.get_best_policy_flag,
                    max_episode_length=self.max_episode_len, observation_length=env.observation_spec_len, 
                    orig_env_use_stacked_observations=self.stacked_observations,
                    use_gumbel_softmax=True, seed=self.seed, use_matrices=True)
                extraction_stats_sized = fsc_actor_sized.behavioral_clone_original_policy_to_fsc(
                    buffer, num_epochs=5001, specification_checker=self.specification_checker,
                    environment=env, tf_environment=tf_env, args=None, extraction_stats=None)
                extraction_stats.add_large_fsc_result(size, extraction_stats_sized.extracted_policy_reachabilities[-1], extraction_stats_sized.extracted_policy_rewards[-1])

        
        del buffer
        self.cloned_actor.set_probs_updates()
        # if self.get_best_policy_flag:
        #     self.cloned_actor.load_best_policy()
        fsc, fsc_actions, fsc_updates = BlackBoxExtractor.extract_fsc(self.cloned_actor, env, self.memory_len, 
                                                                               is_one_hot=self.is_one_hot, non_deterministic=self.non_deterministic,
                                                                               family_quotient_numpy=self.family_quotient_numpy,
                                                                               complete_probs=self.complete_probs)
        self.cloned_actor.unset_probs_updates()
        fsc_res = evaluate_policy_in_model(fsc, environment=env, tf_environment=tf_env, max_steps=(self.max_episode_len + 1) * 2)
        extraction_stats.add_fsc_result(fsc_res.reach_probs[-1], fsc_res.returns[-1])

        return fsc, extraction_stats

    def clone_and_generate_fsc_from_policy(self, original_policy : TFPolicy, 
                                           env : EnvironmentWrapperVec = None, 
                                           tf_env : TFPyEnvironment = None) -> tuple[TableBasedPolicy, ExtractionStats]:
        if self.autlearn_extraction:
            return self.aalpy_extraction(
                original_policy, env, tf_env)
        else:
            return self.self_interpretable_extraction(
                original_policy, env, tf_env)


    @staticmethod
    def create_memory_to_tensor_table(compute_memory, memory_size, max_memory):
        memory_to_tensor_table = [compute_memory(
            memory_size, i) for i in range(max_memory)]
        memory_to_tensor_table = tf.convert_to_tensor(
            memory_to_tensor_table, dtype=tf.float32)
        return memory_to_tensor_table
    
    @staticmethod
    def get_det_action_and_update_function(policy: TFPolicy, time_steps, memory_states, 
                                           max_memory, nr_observations, memory_len, base, 
                                           decompute_memory, action_keywords):
            eager = PyTFEagerPolicy(
                policy, use_tf_function=True, batch_time_steps=False)
            fsc_updates = np.zeros((max_memory, nr_observations))
            fsc_actions = np.zeros((max_memory, nr_observations))
            policy_steps = eager.action(time_steps, policy_state=memory_states)
            actions = policy_steps.action.numpy().reshape((max_memory, nr_observations))
            fsc_actions[:, :] = actions
            states = policy_steps.state  # (M*O, L)
            states = tf.math.argmax(states, axis=-1)  # (M*O,)
            states = decompute_memory(
                memory_len, states, base)
            states = tf.reshape(states, (max_memory, nr_observations)).numpy()  # (M, O)
            fsc_updates[:, :] = states
            table_based_policy = TableBasedPolicy(
                policy, fsc_actions, fsc_updates, initial_memory=0, action_keywords=action_keywords)
            return table_based_policy, fsc_actions, fsc_updates


    @staticmethod
    def extract_fsc(policy: TFPolicy, environment: EnvironmentWrapperVec, memory_len: int,
                    is_one_hot: bool = True, non_deterministic=True,
                    family_quotient_numpy: FamilyQuotientNumpy = None,
                    complete_probs = True) -> tuple[TableBasedPolicy, np.ndarray, np.ndarray]:
        # Computes the number of potential combinations of latent memory (3 possible values for each latent memory cell, {-1, 0, 1})
        base = 3
        max_memory = base ** memory_len if not is_one_hot else memory_len
        nr_observations = environment.stormpy_model.nr_observations if family_quotient_numpy is None else len(family_quotient_numpy.observation_to_legal_action_mask)
        
        
        logger.info("Computing memory to tensor table")
        compute_memory, decompute_memory = get_encoding_functions(is_one_hot, complete_probs=complete_probs)
        memory_to_tensor_table = BlackBoxExtractor.create_memory_to_tensor_table(
            compute_memory, memory_len, max_memory)
        
        logger.info("Starting to extract FSC")

        _, obs_mesh = np.meshgrid(
            np.arange(max_memory), np.arange(nr_observations), indexing='ij')
        obs_mesh = obs_mesh.flatten()
        obs_batch = tf.convert_to_tensor(obs_mesh, dtype=tf.int32)  # (N,)
        if family_quotient_numpy is not None:
            time_steps, illegal_actions_flags = family_quotient_numpy.get_time_steps_for_observation_integers(
                obs_batch, environment.action_keywords)
        else:
            time_steps = environment.create_fake_timestep_from_observation_integer(obs_batch)
            illegal_actions_flags = np.zeros((time_steps.observation["mask"].shape[0], ), dtype=bool)  # No illegal actions in the fake time step
        memory_states = tf.convert_to_tensor(memory_to_tensor_table, dtype=tf.float32)  # (M, L)
        memory_states = tf.reshape(memory_states, (max_memory, 1, memory_len))          # (M, 1, L)
        memory_states = tf.repeat(memory_states, repeats=nr_observations, axis=1)       # (M, O, L)
        memory_states = tf.reshape(memory_states, (-1, memory_len))                 
        if not non_deterministic:    # (M*O, L)
           table_based_policy, fsc_actions, fsc_updates = BlackBoxExtractor.get_det_action_and_update_function(
                policy, time_steps, memory_states, max_memory, nr_observations, memory_len, base, decompute_memory,
                environment.action_keywords)
        else:
            fsc_actions = np.zeros((max_memory, nr_observations, len(environment.action_keywords)), dtype=np.float32)
            distro_function = tf.function(policy.distro)
            logits, states = distro_function(time_steps, policy_state=memory_states, seed=0)
            logits = tf.reshape(logits, (max_memory, nr_observations, len(environment.action_keywords)))  # (M, O, L)
            illegal_actions_flags = np.reshape(illegal_actions_flags, (max_memory, nr_observations))  # (M, O)
            probs = tf.nn.softmax(logits, axis=-1).numpy()  # (M, O, L)
            # Set probs to zero when the action has too low probability
            probs[probs < 0.001] = 0.0
            probs /= np.sum(probs, axis=-1, keepdims=True)  # Normalize probabilities
            # use illegal actions flags to set probabilities to zero to all actions
            probs[illegal_actions_flags, :] = 0.0
            fsc_actions[:, :, :] = probs
            fsc_updates = np.zeros((max_memory, nr_observations, max_memory), dtype=np.float32)
            states = decompute_memory(memory_len, states, base).numpy()  # (M*O) if not complete_probs else (M*O, L)
            if complete_probs:
                states = np.reshape(states, (max_memory, nr_observations, max_memory))
                fsc_updates[:, :, :] = states  # (M, O, L)
            else:
                states = np.reshape(states, (max_memory, nr_observations))  # (M, O)
                # Expand states to match the shape (M, O, L), since updates are deterministic and we want to map state update to dirac distribution
                fsc_updates[:, :, :] = np.eye(max_memory, dtype=np.float32)[states.astype(int)]  # (M, O, L)
            table_based_policy = TableBasedPolicy(
                policy, fsc_actions, fsc_updates, initial_memory=0, action_keywords=environment.action_keywords)

            # states = tf.reshape(states, (max_memory, nr_observations, memory_len))

        return table_based_policy, fsc_actions, fsc_updates

    @staticmethod
    def save_eval_res_to_json(eval_res: EvaluationResults, prism_model: str, path_to_experiment_folder: str):
        if not os.path.exists(path_to_experiment_folder):
            os.makedirs(path_to_experiment_folder)
        index = 0
        while os.path.exists(os.path.join(path_to_experiment_folder, f"{prism_model}_evaluation_results_{index}.json")):
            index += 1
        eval_res.save_to_json(os.path.join(
            path_to_experiment_folder, f"{prism_model}_evaluation_results_{index}.json"))
        
    

    @staticmethod
    def run_benchmark(prism_path : str, properties_path : str, memory_size, num_data_steps=1000, num_training_steps=300,
                       specification_goal="reachability", optimization_goal="max", use_one_hot=False,
                       extraction_epochs=100000, use_residual_connection=False
                       ) -> tuple[ClonedFSCActorPolicy, TFUniformReplayBuffer, ExtractionStats, Recurrent_PPO_agent]:

        args = init_args(prism_path=prism_path, properties_path=properties_path,
                        nr_runs=num_training_steps, goal_value_multiplier=1.00)
        args.save_agent = False
        env, tf_env = init_environment(args)
        model_name = prism_path.split("/")[-2]
        agent = Recurrent_PPO_agent(
            env, tf_env, args, agent_folder=f"{args.agent_name}/{model_name}", load=False)
        agent.train_agent(iterations=num_training_steps)
        specification_checker = SpecificationChecker(
            optimization_specification=specification_goal,
            optimization_goal=optimization_goal,
            evaluation_results=agent.evaluation_result
        )
        # agent.load_agent(True)
        # agent.evaluate_agent(vectorized = True, max_steps = 800)
        extraction_stats = ExtractionStats(
            original_policy_reachability=agent.evaluation_result.reach_probs[-1],
            original_policy_reward=agent.evaluation_result.returns[-1],
            use_one_hot=use_one_hot,
            number_of_samples=num_data_steps * args.num_environments,
            memory_size=memory_size,
            residual_connection=use_residual_connection
        )

        split_path = prism_path.split("/")
        model_name = split_path[-2]
        agent.set_agent_greedy()
        agent.set_policy_masking()
        buffer = sample_data_with_policy(
            agent.collect_policy_wrapper, num_samples=num_data_steps,
            environment=env, tf_environment=tf_env,
        )
        cloned_actor = ClonedFSCActorPolicy(
            agent.wrapper, memory_size, agent.wrapper.observation_and_action_constraint_splitter,
            use_one_hot=use_one_hot, use_residual_connection=use_residual_connection, observation_length=env.observation_spec_len, seed=args.seed)
        # Train the cloned actor (cloned_actor.fsc_actor) to mimic the original policy
        extraction_stats = cloned_actor.behavioral_clone_original_policy_to_fsc(
            buffer, num_epochs=extraction_epochs,
            specification_checker=specification_checker,
            environment=env, tf_environment=tf_env, args=args,
            extraction_stats=extraction_stats
        )
        cloned_actor.set_probs_updates()
        fsc, action_table, update_table = BlackBoxExtractor.extract_fsc(cloned_actor, env, memory_size, is_one_hot=use_one_hot)
        if DEBUG:
            fsc2, action_table2, update_table2 = BlackBoxExtractor.extract_fsc_nonvectorized(cloned_actor, env, memory_size, is_one_hot=use_one_hot)
            # Compare action and update tables (they should be same)
            try:
                assert np.array_equal(action_table, action_table2)
            except AssertionError:
                print("Action tables are not equal")
                print(f"Action table: {action_table}")
                print(f"Action table2: {action_table2}")
            try:
                assert np.array_equal(update_table, update_table2)
            except AssertionError:
                print("Update tables are not equal")
                print(f"Update table: {update_table}")
                print(f"Update table2: {update_table2}")
        
        ev_res = evaluate_policy_in_model(
            fsc, args, env, tf_env, args.max_steps + 1, None)
        extraction_stats.add_fsc_result(ev_res.reach_probs[-1], ev_res.returns[-1])
        return cloned_actor, buffer, extraction_stats, agent
    
def test_sameness_of_extracted_fsc(cloned_actor, env, memory_size, is_one_hot=False, tf_env=None, agent=None):
    fsc, action_table, update_table = BlackBoxExtractor.extract_fsc(cloned_actor, env, memory_size, is_one_hot=is_one_hot)
    fsc2, action_table2, update_table2 = BlackBoxExtractor.extract_fsc_nonvectorized(cloned_actor, env, memory_size, is_one_hot=is_one_hot)
    # Compare action and update tables (they should be same)
    try:
        assert np.array_equal(action_table, action_table2)
    except AssertionError:
        print("Action tables are not equal")
        print(f"Action table: {action_table}")
        print(f"Action table2: {action_table2}")
    try:
        assert np.array_equal(update_table, update_table2)
    except AssertionError:
        print("Update tables are not equal")
        print(f"Update table: {update_table}")
        print(f"Update table2: {update_table2}")
    if DEBUG:
            buffer_test = sample_data_with_policy(
                cloned_actor, num_samples=400, environment=env, tf_environment=tf_env)
            memory_encode, memory_decode = get_encoding_functions(is_one_hot)
            compare_two_policies(cloned_actor, fsc, buffer_test, memory_encode,
                                memory_decode, memory_size, agent.environment)


def parse_args_from_cmd():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prism-path", type=str, required=True)
    parser.add_argument("--memory-size", type=int, default=2)
    parser.add_argument("--num-data-steps", type=int, default=4001)
    parser.add_argument("--num-training-steps", type=int, default=101)
    parser.add_argument("--specification-goal",
                        type=str, default="reachability")
    parser.add_argument("--optimization-goal", type=str, default="max")
    parser.add_argument("--use-one-hot", action="store_true")
    parser.add_argument("--experiments-storage-path-folder",
                        type=str, default="experiments_extraction")
    parser.add_argument("--extraction-epochs", type=int, default=20001)
    parser.add_argument("--use-residual-connection", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args_from_cmd()
    # test_memory_endoce_and_decode_functions(encode, decode, max_memory, memory_size)
    prism_templ = os.path.join(args.prism_path, "sketch.templ")
    properties_templ = os.path.join(args.prism_path, "sketch.props")
    _, _, extraction_stats, og_agent = BlackBoxExtractor.run_benchmark(prism_templ, properties_templ, args.memory_size, 
                                                                     args.num_data_steps, args.num_training_steps,
                                                                     args.specification_goal, args.optimization_goal, args.use_one_hot,
                                                                     args.extraction_epochs, args.use_residual_connection)
    
    extraction_stats.store_as_json(args.prism_path.split(
        "/")[-1], args.experiments_storage_path_folder)
    BlackBoxExtractor.save_eval_res_to_json(og_agent.evaluation_result, args.prism_path.split(
        "/")[-1], args.experiments_storage_path_folder)
