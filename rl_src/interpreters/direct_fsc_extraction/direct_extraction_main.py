from keras import layers, models, optimizers
from keras import backend as K
import numpy as np

import tensorflow as tf
import keras

from interpreters.direct_fsc_extraction.test_functions import *
from interpreters.direct_fsc_extraction.encoding_functions import get_encoding_functions

from tf_agents.trajectories import StepType, Trajectory
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.policies import TFPolicy
from tf_agents.specs import BoundedArraySpec
from tf_agents.environments import TFPyEnvironment
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy
from tf_agents.trajectories import TimeStep

import tf_agents

from environment.environment_wrapper_vec import EnvironmentWrapperVec

from rl_src.tests.general_test_tools import init_environment, init_args
from rl_src.agents.recurrent_ppo_agent import Recurrent_PPO_agent
from rl_src.tools.evaluators import evaluate_policy_in_model, EvaluationResults

from rl_src.interpreters.bottlenecking.quantized_bottleneck_extractor import TableBasedPolicy
from rl_src.tools.args_emulator import ArgsEmulator
from rl_src.tools.specification_check import SpecificationChecker

import logging

import os

from rl_src.interpreters.direct_fsc_extraction.extraction_stats import ExtractionStats
from interpreters.direct_fsc_extraction.fsc_like_actor_network import FSCLikeActorNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClonedFSCActorPolicy(TFPolicy):
    def __init__(self, original_policy: TFPolicy, memory_size: int, 
                 observation_and_action_constrint_splitter=None, 
                 use_one_hot=True,
                 use_residual_connection=True):
        self.original_policy = original_policy
        self.use_one_hot = use_one_hot
        policy_state_spec = BoundedArraySpec(
            (memory_size,), np.float32, minimum=-1.5, maximum=1.5)
        super(ClonedFSCActorPolicy, self).__init__(time_step_spec=original_policy.time_step_spec,
                                                   action_spec=original_policy.action_spec,
                                                   policy_state_spec=policy_state_spec,
                                                   observation_and_action_constraint_splitter=observation_and_action_constrint_splitter)
        self.memory_size = memory_size
        self.fsc_actor = FSCLikeActorNetwork(
            original_policy.time_step_spec.observation["observation"].shape,
            original_policy.action_spec.maximum + 1,
            memory_size,
            use_one_hot=use_one_hot,
            use_residual_connection=use_residual_connection)

    def _variables(self):
        return self.fsc_actor.variables

    def distro(self, time_step, policy_state, seed):
        observation, mask = self.observation_and_action_constraint_splitter(
            time_step.observation)
        observation = tf.reshape(observation, (observation.shape[0], 1, -1))
        policy_state = tf.reshape(policy_state, (policy_state.shape[0], -1))
        step_type = tf.reshape(time_step.step_type,
                               (time_step.step_type.shape[0], 1, -1))
        step_type = tf.cast(step_type, tf.float32)
        action, memory = self.fsc_actor(
            observation, step_type, policy_state)
        action = tf.reshape(action, (action.shape[0], -1))
        # Change logits of illegal actions to -inf
        action = tf.where(mask, action, -1e20)
        # memory = memory[:, -1, :]
        return action, memory

    def _action(self, time_step, policy_state, seed):
        action_probs, memory = self.distro(
            time_step, policy_state, seed)
        # action = tf.random.categorical(action_probs, 1, dtype=tf.int32)
        # sample the most probable action
        action = tf.argmax(action_probs, axis=-1, output_type=tf.int32)
        action = tf.reshape(action, (action.shape[0],))
        policy_step = PolicyStep(action=action, state=memory)
        # print(policy_step)
        return policy_step

    def _get_initial_state(self, batch_size: int):
        init_state = self.fsc_actor.get_initial_state(batch_size)
        return tf.reshape(init_state, (batch_size, -1))
    
    
    def behavioral_clone_original_policy_to_fsc(self, buffer: TFUniformReplayBuffer, num_epochs: int,
                                            sample_len=25,
                                            specification_checker: SpecificationChecker = None,
                                            environment: EnvironmentWrapperVec = None,
                                            tf_environment: TFPyEnvironment = None,
                                            args: ArgsEmulator = None,
                                            extraction_stats: ExtractionStats = None) -> ExtractionStats:
        cloned_actor = self
        dataset_options = tf.data.Options()
        dataset_options.experimental_deterministic = False
        dataset = (
            buffer.as_dataset(sample_batch_size=64, num_steps=sample_len,
                            num_parallel_calls=tf.data.AUTOTUNE)
            .with_options(dataset_options)
            .prefetch(tf.data.AUTOTUNE)
        )

        if extraction_stats is None:
            extraction_stats = ExtractionStats(
                original_policy_reachability=0,
                original_policy_reward=0,
                use_one_hot=self.use_one_hot,
                number_of_samples=sample_len,
                memory_size=self.memory_size,
                residual_connection=self.fsc_actor.use_residual_connection
            )

        iterator = iter(dataset)
        neural_fsc = cloned_actor.fsc_actor
        optimizer = optimizers.Adam(learning_rate=1.6e-4)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss_metric = tf.keras.metrics.Mean(name="train_loss")
        accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(
            name="accuracy")
        
        evaluation_result = None

        @tf.function
        def train_step(experience):
            observations = experience.observation["observation"]
            gt_actions = experience.action
            step_types = tf.cast(experience.step_type, tf.float32)
            step_types = tf.reshape(step_types, (step_types.shape[0], -1, 1))
            with tf.GradientTape() as tape:
                played_action, mem = neural_fsc(observations, step_type=step_types)
                loss = loss_fn(gt_actions, played_action)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, neural_fsc.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(zip(grads, neural_fsc.trainable_variables))
            loss_metric.update_state(loss)
            accuracy_metric.update_state(gt_actions, played_action
                                        )
            return loss

        for i in range(num_epochs):
            try:
                experience, _ = next(iterator)
            except StopIteration:  # Reset iteratoru, pokud dojde dataset
                iterator = iter(dataset)
                experience, _ = next(iterator)

            loss = train_step(experience)
            self.periodical_evaluation(i, loss_metric, accuracy_metric, cloned_actor, args, 
                                       environment, tf_environment, extraction_stats, 
                                       evaluation_result, specification_checker)
            

        return extraction_stats
    
    def periodical_evaluation(self, i, loss_metric, accuracy_metric, cloned_actor, args, 
                              environment, tf_environment, extraction_stats, 
                              evaluation_result, specification_checker):
        if i % 1000 == 0:
                avg_loss = loss_metric.result()
                accuracy = accuracy_metric.result()
                logger.info(f"Epoch {i}, Loss: {avg_loss:.4f}")
                logger.info(f"Epoch {i}, Accuracy: {accuracy:.4f}")
                extraction_stats.add_evaluation_accuracy(accuracy)
                loss_metric.reset_states()
                accuracy_metric.reset_states()
        if i % 5000 == 0:
                evaluation_result = evaluate_policy_in_model(
                    cloned_actor, args, environment, tf_environment, args.max_steps + 1, evaluation_result)
                extraction_stats.add_extraction_result(
                    evaluation_result.reach_probs[-1], evaluation_result.returns[-1])
                if False and specification_checker is not None:
                    if specification_checker.check_specification(evaluation_result.reach_probs[-1], evaluation_result.returns[-1]):
                        pass


def sample_data_with_policy(policy: TFPolicy, num_samples=100,
                            environment: EnvironmentWrapperVec = None,
                            tf_environment: TFPyEnvironment = None) -> TFUniformReplayBuffer:
    prev_time_step = tf_environment.reset()
    policy_state = policy.get_initial_state(environment.batch_size)
    replay_buffer = TFUniformReplayBuffer(
        data_spec=policy.trajectory_spec, batch_size=environment.batch_size, max_length=num_samples+1)
    action_function = tf_agents.utils.common.function(policy.action)

    for i in range(num_samples):
        policy_step = action_function(prev_time_step, policy_state)
        action = policy_step.action
        policy_state = policy_step.state
        time_step = tf_environment.step(action)
        traj = tf_agents.trajectories.trajectory.from_transition(
            prev_time_step, policy_step, time_step)
        replay_buffer.add_batch(traj)
        prev_time_step = time_step
    return replay_buffer


def create_memory_to_tensor_table(compute_memory, memory_size, max_memory):
    memory_to_tensor_table = [compute_memory(
        memory_size, i) for i in range(max_memory)]
    memory_to_tensor_table = tf.convert_to_tensor(
        memory_to_tensor_table, dtype=tf.float32)
    return memory_to_tensor_table


def extract_fsc(policy: TFPolicy, environment: EnvironmentWrapperVec, memory_len: int, is_one_hot: bool = True) -> TableBasedPolicy:
    # Computes the number of potential combinations of latent memory (3 possible values for each latent memory cell, {-1, 0, 1})
    base = 3
    max_memory = base ** memory_len if not is_one_hot else memory_len
    nr_observations = environment.stormpy_model.nr_observations
    fsc_actions = np.zeros((max_memory, nr_observations))
    fsc_updates = np.zeros((max_memory, nr_observations))

    compute_memory, decompute_memory = get_encoding_functions(is_one_hot)
    memory_to_tensor_table = create_memory_to_tensor_table(
        compute_memory, memory_len, max_memory)
    eager = PyTFEagerPolicy(
        policy, use_tf_function=True, batch_time_steps=False)

    for i in range(nr_observations):
        # Go thrgough all memory permutations
        fake_time_step = environment.create_fake_timestep_from_observation_integer(
            i)
        for j in range(max_memory):
            policy_state = memory_to_tensor_table[j]
            policy_state = tf.reshape(policy_state, (1, memory_len))
            policy_step = eager.action(
                fake_time_step, policy_state=policy_state)
            
            fsc_actions[j, i] = policy_step.action.numpy()[0]
            fsc_updates[j, i] = decompute_memory(
                memory_len, policy_step.state, base)

    table_based_policy = TableBasedPolicy(
        policy, fsc_actions, fsc_updates, initial_memory=0)
    return table_based_policy

def save_eval_res_to_json(eval_res: EvaluationResults, prism_model: str, path_to_experiment_folder: str):
    if not os.path.exists(path_to_experiment_folder):
        os.makedirs(path_to_experiment_folder)
    index = 0
    while os.path.exists(os.path.join(path_to_experiment_folder, f"{prism_model}_evaluation_results_{index}.json")):
        index += 1
    eval_res.save_to_json(os.path.join(path_to_experiment_folder, f"{prism_model}_evaluation_results_{index}.json"))



        
def run_experiment(prism_path, properties_path, memory_size, num_data_steps=100, num_training_steps=300,
                   specification_goal="reachability", optimization_goal="max", use_one_hot=False,
                   extraction_epochs=100000, use_residual_connection=False) -> tuple[ClonedFSCActorPolicy, TFUniformReplayBuffer, ExtractionStats, Recurrent_PPO_agent]:

    args = init_args(prism_path=prism_path, properties_path=properties_path,
                     nr_runs=num_training_steps, goal_value_multiplier=1.00)
    args.agent_name = "FSC_Clone"
    args.save_agent = True
    env, tf_env = init_environment(args)
    agent = Recurrent_PPO_agent(
        env, tf_env, args, agent_folder=args.agent_name, load=False)
    agent.train_agent(iterations=num_training_steps)
    specification_checker = SpecificationChecker(
        optimization_specification=specification_goal,
        optimization_goal=optimization_goal,
        evaluation_results=agent.evaluation_result
    )
    extraction_stats = ExtractionStats(
        original_policy_reachability=agent.evaluation_result.reach_probs[-1],
        original_policy_reward=agent.evaluation_result.returns[-1],
        use_one_hot=use_one_hot,
        number_of_samples=num_data_steps * args.num_environments,
        memory_size=memory_size,
        residual_connection=use_residual_connection
    )

    # agent.load_agent(True)
    # agent.evaluate_agent(vectorized = True, max_steps = 800)

    split_path = prism_path.split("/")
    model_name = split_path[-2]
    agent.set_agent_greedy()
    buffer = sample_data_with_policy(
        agent.wrapper, num_samples=num_data_steps,
        environment=env, tf_environment=tf_env,
    )
    cloned_actor = ClonedFSCActorPolicy(
            agent.wrapper, memory_size, agent.wrapper.observation_and_action_constraint_splitter, 
            use_one_hot=use_one_hot, use_residual_connection=use_residual_connection)
    # Train the cloned actor (cloned_actor.fsc_actor) to mimic the original policy
    extraction_stats = cloned_actor.behavioral_clone_original_policy_to_fsc(
        buffer, num_epochs=extraction_epochs,
        specification_checker=specification_checker,
        environment=env, tf_environment=tf_env, args=args,
        extraction_stats=extraction_stats
        )
    fsc = extract_fsc(cloned_actor, env, memory_size, is_one_hot=use_one_hot)
    buffer_test = sample_data_with_policy(
        cloned_actor, num_samples=400, environment=env, tf_environment=tf_env)
    memory_encode, memory_decode = get_encoding_functions(use_one_hot)
    compare_two_policies(cloned_actor, fsc, buffer_test, memory_encode, memory_decode, memory_size, agent.environment)
    ev_res = evaluate_policy_in_model(
        fsc, args, env, tf_env, args.max_steps + 1, None)
    extraction_stats.add_fsc_result(ev_res.reach_probs[-1], ev_res.returns[-1])
    return cloned_actor, buffer, extraction_stats, agent


def parse_args_from_cmd():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prism-path", type=str, required=True)
    parser.add_argument("--memory-size", type=int, default=2)
    parser.add_argument("--num-data-steps", type=int, default=6000)
    parser.add_argument("--num-training-steps", type=int, default=3000)
    parser.add_argument("--specification-goal",
                        type=str, default="reachability")
    parser.add_argument("--optimization-goal", type=str, default="max")
    parser.add_argument("--use-one-hot", action="store_true")
    parser.add_argument("--experiments-storage-path-folder",
                        type=str, default="experiments_extraction")
    parser.add_argument("--extraction-epochs", type=int, default=100000)
    parser.add_argument("--use-residual-connection", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args_from_cmd()
    # test_memory_endoce_and_decode_functions(encode, decode, max_memory, memory_size)
    prism_templ = os.path.join(args.prism_path, "sketch.templ")
    properties_templ = os.path.join(args.prism_path, "sketch.props")
    _, _, extraction_stats, og_agent = run_experiment(prism_templ, properties_templ, args.memory_size, args.num_data_steps, args.num_training_steps,
                                            args.specification_goal, args.optimization_goal, args.use_one_hot,
                                            args.extraction_epochs, args.use_residual_connection)
    
    extraction_stats.store_as_json(args.prism_path.split("/")[-1], args.experiments_storage_path_folder)
    save_eval_res_to_json(og_agent.evaluation_result, args.prism_path.split("/")[-1], args.experiments_storage_path_folder)
    
