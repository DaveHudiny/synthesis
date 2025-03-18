from keras import layers, models, optimizers
from keras import backend as K
import numpy as np

import tensorflow as tf
import keras

from tf_agents.trajectories import StepType, Trajectory
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.policies import TFPolicy
from tf_agents.specs import BoundedArraySpec
from tf_agents.environments import TFPyEnvironment
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy

import tf_agents

from environment.environment_wrapper_vec import EnvironmentWrapperVec

from rl_src.tests.general_test_tools import init_environment, init_args
from rl_src.agents.recurrent_ppo_agent import Recurrent_PPO_agent
from rl_src.tools.evaluators import evaluate_policy_in_model, EvaluationResults

from rl_src.interpreters.bottlenecking.quantized_bottleneck_extractor import TableBasedPolicy
from rl_src.tools.args_emulator import ArgsEmulator
from rl_src.tools.specification_check import SpecificationChecker

import logging

import os, sys
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtractionStats:
    def __init__(self, original_policy_reachability : float, original_policy_reward : float,
                 use_one_hot: bool = False, number_of_samples: int = 0, memory_size : int = 0):
        
        self.original_policy_reachability = original_policy_reachability
        self.original_policy_reward = original_policy_reward
        self.use_one_hots = use_one_hot
        self.memory_size = memory_size
        self.number_of_samples = number_of_samples

        self.extracted_policy_reachabilities = []
        self.extracted_policy_rewards = []
        self.extracted_fsc_reachability = None
        self.extracted_fsc_reward = None

        self.evaluation_accuracies = []

    def add_extraction_result(self, extracted_policy_reachability : float, extracted_policy_reward : float):
        self.extracted_policy_reachabilities.append(extracted_policy_reachability)
        self.extracted_policy_rewards.append(extracted_policy_reward)

    def add_fsc_result(self, extracted_fsc_reachability : float, extracted_fsc_reward : float):
        self.extracted_fsc_reachability.append(extracted_fsc_reachability)
        self.extracted_fsc_reward.append(extracted_fsc_reward)

    def add_evaluation_accuracy(self, evaluation_accuracy : float):
        self.evaluation_accuracies.append(evaluation_accuracy)

    def store_as_json(self, model_name : str, experiments_path : str):
        if not os.path.exists(experiments_path):
            os.makedirs(experiments_path)
        index = 0
        while os.path.exists(os.path.join(experiments_path, f"{model_name}_extraction_stats_{index}.json")):
            index += 1
        with open(os.path.join(experiments_path, f"{model_name}_extraction_stats_{index}.json"), "w") as f:
            for key, value in self.__dict__.items():
                if isinstance(value, list):
                    value = [float(v) for v in value]
                f.write(f"{key}: {value}\n")

class NeuralFSCActor(models.Model):
    def __init__(self, observation_shape : tf.TensorShape, action_range : int, memory_len : int, use_one_hot : bool =False, use_residual_connection : bool = True):
        super(NeuralFSCActor, self).__init__()
        self.observation_shape = observation_shape
        self.action_range = action_range
        self.memory_len = memory_len
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.simple_rnn_for_memory = layers.SimpleRNN(
            memory_len, return_sequences=True, return_state=True)
        self.projection_network = layers.Dense(64, activation='relu')
        self.action = layers.Dense(self.action_range, activation=None)
        if not use_one_hot:
            self.memory_function = layers.Lambda(
                lambda x: 1.5 * tf.tanh(x) + 0.5 * tf.tanh(-3 * x))
            self.quantization_layer = layers.Lambda(lambda x: tf.round(x))
            self.one_hot_constant = 0
        else:
            self.memory_function = layers.Lambda(
                lambda x: tf.keras.activations.sigmoid(x))
            self.quantization_layer = layers.Lambda(lambda x: tf.one_hot(tf.argmax(x, axis=-1),
                                                                         depth=self.memory_len, dtype=tf.float32))
            self.one_hot_constant = 1
        self.use_residual_connection = use_residual_connection

    def get_initial_state(self, batch_size):
        zeros_like = tf.zeros(
            (batch_size, 1, self.memory_len - self.one_hot_constant))
        return tf.concat([tf.ones((batch_size, 1, self.one_hot_constant)), zeros_like], axis=-1)

    @tf.function
    def call(self, inputs, step_type: StepType, old_memory=None):
        # inputs = tf.concat([inputs, tf.cast(old_memory, tf.float32)], axis=-1)
        inputs = tf.concat([inputs], axis=-1)
        x = self.dense1(inputs)

        if self.use_residual_connection:
            x1, memory = self.simple_rnn_for_memory(x, initial_state=old_memory)
            x2 = self.projection_network(x1)
            x = layers.concatenate([x1, x2], axis=-1)
        else:
            x = self.dense2(x)
            x, memory = self.simple_rnn_for_memory(x, initial_state=old_memory)

        # x2 = self.projection_network(x)
        # x = layers.concatenate(x1, axis=-1)
        
        action = self.action(x)
        # State-through estimation, where we ignore round(x)
        # memory = keras.activations.sigmoid(memory)
        memory = self.memory_function(memory)
        x_quantized = self.quantization_layer(memory)
        memory = memory + tf.stop_gradient(x_quantized - memory)
        return action, memory


class ClonedFSCActorPolicy(TFPolicy):
    def __init__(self, original_policy: TFPolicy, memory_size: int, observation_and_action_constrint_splitter=None, use_one_hot=True):
        self.original_policy = original_policy
        self.use_one_hot = use_one_hot
        policy_state_spec = BoundedArraySpec(
            (memory_size,), np.float32, minimum=-1.5, maximum=1.5)
        super(ClonedFSCActorPolicy, self).__init__(time_step_spec=original_policy.time_step_spec,
                                                   action_spec=original_policy.action_spec,
                                                   policy_state_spec=policy_state_spec,
                                                   observation_and_action_constraint_splitter=observation_and_action_constrint_splitter)
        self.memory_size = memory_size
        self.fsc_actor = NeuralFSCActor(
            original_policy.time_step_spec.observation["observation"].shape, 
            original_policy.action_spec.maximum + 1, 
            memory_size,
            use_one_hot=use_one_hot)

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
        return action, memory

    def _action(self, time_step, policy_state, seed):
        action_probs, memory = self.distro(
            time_step, policy_state, seed)
        # action = tf.random.categorical(action_probs, 1, dtype=tf.int32)
        # sample the most probable action
        action = tf.argmax(action_probs, axis=-1, output_type=tf.int32)
        action = tf.reshape(action, (action.shape[0],))
        policy_step = PolicyStep(action=action, state=memory)
        return policy_step

    def _get_initial_state(self, batch_size : int):
        init_state = self.fsc_actor.get_initial_state(batch_size)
        return tf.reshape(init_state, (batch_size, -1))


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


def compute_rounded_memory(memory_size: int, memory_int: int, memory_base=3) -> tf.Tensor:
    memory = np.zeros((memory_size,))
    # increase the memory by 1 given the previous memory. Every memory cell can be {-1, 0, 1}
    for i in range(memory_size):
        memory[i] = ((memory_int + 1) % memory_base) - 1
        memory_int = memory_int // memory_base
    return tf.convert_to_tensor(memory, dtype=tf.float32)


def decompute_rounded_memory(memory_size: int, memory_vector: tf.Tensor, memory_base = 3) -> int:
    memory_int = 0
    for i in range(memory_size):
        memory_int += (memory_vector[0][i] + 1) * (memory_base ** i)
    return memory_int

def one_hot_encode_memory(memory_size: int, memory_int : int, memory_base=3) -> tf.Tensor:
    memory = np.zeros((memory_size,))
    memory[memory_int] = 1
    return tf.convert_to_tensor(memory, dtype=tf.float32)

def one_hot_decode_memory(memory_size = 0, memory_vector: tf.Tensor = None, memory_base = 3) -> int:
    return tf.argmax(memory_vector, axis=-1).numpy()[0]

def get_encoding_functions(is_one_hot: bool = True) -> tuple[callable, callable]:
    if is_one_hot:
        compute_memory = one_hot_encode_memory
        decompute_memory = one_hot_decode_memory
    else:
        compute_memory = compute_rounded_memory
        decompute_memory = decompute_rounded_memory
    return compute_memory, decompute_memory

def create_memory_to_tensor_table(compute_memory, memory_size, max_memory):
    memory_to_tensor_table = [compute_memory(
        memory_size, i) for i in range(max_memory)]
    memory_to_tensor_table = tf.convert_to_tensor(
        memory_to_tensor_table, dtype=tf.float32)
    return memory_to_tensor_table

def extract_fsc(policy: TFPolicy, environment: EnvironmentWrapperVec, memory_len: int, args: ArgsEmulator, is_one_hot : bool = True) -> TableBasedPolicy:
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
            policy_step = eager.action(
                fake_time_step, policy_state=policy_state)
            fsc_actions[j, i] = policy_step.action.numpy()[0]
            fsc_updates[j, i] = decompute_memory(
                memory_len, policy_step.state, base)

    table_based_policy = TableBasedPolicy(
        policy, fsc_actions, fsc_updates, initial_memory=0)
    return table_based_policy

def behavioral_clone_original_policy_to_fsc(buffer: TFUniformReplayBuffer, num_epochs: int, policy: TFPolicy,
                                            memory_size: int, sample_len=25, optimal_value = 0.0,
                                            specification_checker : SpecificationChecker = None,
                                            observation_and_action_constraint_splitter : callable = lambda x: x,
                                            environment: EnvironmentWrapperVec = None,
                                            tf_environment: TFPyEnvironment = None,
                                            args: ArgsEmulator = None,
                                            use_one_hot = False) -> tuple[ClonedFSCActorPolicy, ExtractionStats]:
    cloned_actor = ClonedFSCActorPolicy(
        policy, memory_size, observation_and_action_constraint_splitter, use_one_hot=use_one_hot)
    dataset_options = tf.data.Options()
    dataset_options.experimental_deterministic = False
    dataset = (
        buffer.as_dataset(sample_batch_size=64, num_steps=sample_len,
                          num_parallel_calls=tf.data.AUTOTUNE)
        .with_options(dataset_options)
        .prefetch(tf.data.AUTOTUNE)
    )

    iterator = iter(dataset)
    neural_fsc = cloned_actor.fsc_actor
    optimizer = optimizers.Adam(learning_rate=1.6e-4)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_metric = tf.keras.metrics.Mean(name="train_loss")
    accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

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

        if i % 1000 == 0:
            avg_loss = loss_metric.result()
            accuracy = accuracy_metric.result()
            logger.info(f"Epoch {i}, Loss: {avg_loss:.4f}")
            logger.info(f"Epoch {i}, Accuracy: {accuracy:.4f}")

            loss_metric.reset_states()
            accuracy_metric.reset_states()
            # Evaluate accuracy of predicted actions on provided dataset
            pass
        if i % 5000 == 0:
            evaluation_result = evaluate_policy_in_model(
                cloned_actor, args, environment, tf_environment, args.max_steps + 1, evaluation_result)

    return cloned_actor


def run_experiment(prism_path, properties_path, memory_size, num_data_steps=100, num_training_steps=300, 
                   specification_goal = "reachability", optimization_goal = "max", use_one_hot = False,
                   experiments_storage_path_folder = "experiments_extraction",
                   extraction_epochs = 100000):

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
    # agent.load_agent(True)
    # agent.evaluate_agent(vectorized = True, max_steps = 800)

    split_path = prism_path.split("/")
    model_name = split_path[-2]
    agent.set_agent_greedy()
    buffer = sample_data_with_policy(
        agent.wrapper, num_samples=num_data_steps, 
        environment=env, tf_environment=tf_env,
        )
    # Train the cloned actor (cloned_actor.fsc_actor) to mimic the original policy
    cloned_actor, extraction_stats = behavioral_clone_original_policy_to_fsc(
        buffer, num_epochs=extraction_epochs, policy=agent.wrapper, 
        memory_size=memory_size, 
        specification_checker=specification_checker,
        observation_and_action_constraint_splitter=agent.wrapper.observation_and_action_constraint_splitter,
        environment=env, tf_environment=tf_env, args=args, use_one_hot=use_one_hot)
    fsc = extract_fsc(cloned_actor, env, memory_size, args, is_one_hot=use_one_hot)
    evaluate_policy_in_model(fsc, args, env, tf_env, args.max_steps + 1, None)
    return cloned_actor, buffer

def parse_args_from_cmd():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prism-path", type=str, required=True)
    parser.add_argument("--properties-path", type=str, required=True)
    parser.add_argument("--memory-size", type=int, default=2)
    parser.add_argument("--num-data-steps", type=int, default=5000)
    parser.add_argument("--num-training-steps", type=int, default=3)
    parser.add_argument("--specification-goal", type=str, default="reachability")
    parser.add_argument("--optimization-goal", type=str, default="max")
    parser.add_argument("--use-one-hot", action="store_true")
    parser.add_argument("--experiments-storage-path-folder", type=str, default="experiments_extraction")
    parser.add_argument("--extraction-epochs", type=int, default=1000)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # run_experiment("models_paynt_experiments/geo-2-8/sketch.templ",
    #                "models_paynt_experiments/geo-2-8/sketch.props", 2, 6000, 1000,
    #                specification_goal="reachability", optimization_goal="max")

    args = parse_args_from_cmd()
    run_experiment(args.prism_path, args.properties_path, args.memory_size, args.num_data_steps, args.num_training_steps,
                   args.specification_goal, args.optimization_goal, args.use_one_hot, args.experiments_storage_path_folder,
                     args.extraction_epochs)