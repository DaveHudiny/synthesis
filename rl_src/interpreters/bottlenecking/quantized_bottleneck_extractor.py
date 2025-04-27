import tensorflow as tf
from keras import layers, models
import numpy as np

from tf_agents.policies import TFPolicy
from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy
from tf_agents.trajectories.policy_step import PolicyStep

from interpreters.bottlenecking.bottleneck_autoencoder import Encoder, Decoder, Autoencoder

from environment.tf_py_environment import TFPyEnvironment
from agents.father_agent import FatherAgent

from tools.evaluation_results_class import EvaluationResults
from tools.evaluators import *
from tools.saving_tools import *

from rl_src.tests.general_test_tools import *
from agents.recurrent_ppo_agent import Recurrent_PPO_agent
from environment.environment_wrapper_vec import EnvironmentWrapperVec

from interpreters.bottlenecking.bottlenecked_actor_network import BottleneckedActor

from interpreters.extracted_fsc.table_based_policy import TableBasedPolicy

import sys
import os


class AutoencodedPolicy(TFPolicy):
    def __init__(self, policy: TFPolicy, autoencoder: Autoencoder):
        self.policy = policy
        self.autoencoder = autoencoder
        # Initialize policy super class
        super(AutoencodedPolicy, self).__init__(policy.time_step_spec,
                                                policy.action_spec, policy.policy_state_spec, policy.info_spec)

    def _action(self, time_step, policy_state, seed):
        action_step = self.policy.action(time_step, policy_state=policy_state)
        policy_state = action_step.state
        for state_part1 in policy_state:
            substates = []
            for state_part2 in policy_state[state_part1]:
                substates.append(state_part2 if isinstance(
                    state_part2, tf.Tensor) else tf.convert_to_tensor(state_part2))
            substates = tf.concat(substates, axis=-1)
            substates = self.autoencoder(substates)
            # Deconcatenate substates
            new_state = tf.split(substates, num_or_size_splits=2, axis=-1)
            policy_state[state_part1] = new_state
        return PolicyStep(action=action_step.action, state=policy_state, info=action_step.info)

class BottleneckExtractor:

    def __init__(self, tf_environment: TFPyEnvironment, input_dim, latent_dim):
        self.tf_environment = tf_environment
        self.autoencoder = Autoencoder(input_dim, latent_dim, 16)
        self.latent_dim = latent_dim
        self.dataset = []

    def create_generator(self, policy: TFPolicy) -> iter:
        """ Creates tensorflow generator from dataset, that produces batches of data, where x and y are the same.
        """
        self.tf_dataset = tf.data.Dataset.from_tensor_slices(
            self.dataset).batch(64).repeat()
        return iter(zip(self.tf_dataset, self.tf_dataset))

    def collect_data(self, num_data_steps: int, policy: TFPolicy):
        eager = PyTFEagerPolicy(
            policy, use_tf_function=True, batch_time_steps=False)
        self.tf_environment.reset()
        policy_state = policy.get_initial_state(self.tf_environment.batch_size)
        for step in range(num_data_steps):
            time_step = self.tf_environment.current_time_step()
            action_step = eager.action(time_step, policy_state=policy_state)
            next_time_step = self.tf_environment.step(action_step.action)
            policy_state = action_step.state
            for state_part1 in policy_state:
                substates = []
                for state_part2 in policy_state[state_part1]:
                    substates.append(state_part2.numpy())
                substates = tf.concat(substates, axis=-1)
                self.dataset.append(substates)
        self.dataset = np.array(self.dataset)
        self.dataset = np.concatenate(self.dataset, axis=0)

    def evaluate_bottlenecking(self, agent: FatherAgent, max_steps : int = None) -> EvaluationResults:
        bottlenecked_policy = AutoencodedPolicy(
            agent.wrapper, self.autoencoder)
        if max_steps is None:
            max_steps = agent.args.max_steps + 1 
        evaluation_result = evaluate_policy_in_model(
            bottlenecked_policy, agent.args, agent.environment, agent.tf_environment, max_steps = max_steps)

        return evaluation_result
        eager = PyTFEagerPolicy(
            policy, use_tf_function=True, batch_time_steps=False)
        self.tf_environment.reset()
        policy_state = policy.get_initial_state(self.tf_environment.batch_size)
        for step in range(1000):
            time_step = self.tf_environment.current_time_step()
            action_step = eager.action(time_step, policy_state=policy_state)
            next_time_step = self.tf_environment.step(action_step.action)
            policy_state = action_step.state
            for state_part1 in policy_state:
                substates = []
                for state_part2 in policy_state[state_part1]:
                    substates.append(state_part2.numpy())
                substates = tf.concat(substates, axis=-1)
                substates = self.autoencoder(substates)
                # Deconcatenate substates
                new_state = tf.split(substates, num_or_size_splits=2, axis=-1)
                policy_state[state_part1] = new_state

            # Evaluate number of final time_steps and their reward

    def train_autoencoder(self, policy: TFPolicy, num_epochs, num_data_steps):
        self.collect_data(num_data_steps, policy)
        generator = self.create_generator(policy)

        self.autoencoder.compile(optimizer='adam', loss='mse')
        print("Training autoencoder")
        # print sample of data
        self.autoencoder.fit(generator, epochs=num_epochs, steps_per_epoch=500)

    @staticmethod
    def convert_memory_number_to_vector(memory_number: int, num_of_mem_cells: int, base: int = 3) -> np.array:
        """Converts memory number to vector representation, where each value is from set {-1, 0, 1}"""
        memory_vector = np.zeros(num_of_mem_cells)
        for i in range(num_of_mem_cells):
            memory_vector[i] = (memory_number % base) - 1
            memory_number //= base
        return tf.constant(memory_vector, dtype=tf.float32)
    
    def convert_memory_vector_to_number(memory_vector: np.array, base: int = 3) -> int:
        memory_number = 0
        memory_vector = np.squeeze(memory_vector)
        if memory_vector.shape == (): # If the memory vector is a scalar
            memory_vector = np.array([memory_vector])
        for i in range(len(memory_vector)):
            memory_number += (memory_vector[i] + 1) * (base ** i)
        return memory_number

    def extract_fsc(self, policy: TFPolicy, environment: EnvironmentWrapperVec):
        # Computes the number of potential combinations of latent memory (3 possible values for each latent memory cell, {-1, 0, 1})
        memory_size = 3 ** self.latent_dim
        nr_observations = environment.stormpy_model.nr_observations
        fsc_actions = np.zeros((memory_size, nr_observations))
        fsc_updates = np.zeros((memory_size, nr_observations))
        eager = PyTFEagerPolicy(
            policy, use_tf_function=True, batch_time_steps=False)

        decode = tf.function(self.autoencoder.decode)
        encode = tf.function(self.autoencoder.encode)

        initial_state = eager.get_initial_state(1)
        state_name = list(initial_state.keys())[0]

        encoded_initial_state = encode(tf.concat(initial_state[state_name], axis=-1))
        initial_memory = BottleneckExtractor.convert_memory_vector_to_number(encoded_initial_state)
        for i in range(nr_observations):
            # Go thrgough all memory permutations
            fake_time_step = environment.create_fake_timestep_from_observation_integer(
                i)
            for j in range(memory_size):
                memory_vector = BottleneckExtractor.convert_memory_number_to_vector(
                    j, self.latent_dim)
                memory_vector = tf.expand_dims(memory_vector, axis=0)
                decoded_memory = decode(memory_vector)
                policy_state = {state_name: tf.split(
                    decoded_memory, num_or_size_splits=2, axis=-1)}
                policy_step = eager.action(
                    fake_time_step, policy_state=policy_state)
                
                fsc_actions[j, i] = policy_step.action.numpy()[0]
                
                concatenated_state = tf.concat(policy_step.state[state_name], axis=-1)
                encoded_memory = encode(concatenated_state)
                fsc_updates[j, i] = BottleneckExtractor.convert_memory_vector_to_number(encoded_memory)
        return TableBasedPolicy(policy, fsc_actions, fsc_updates, initial_memory = initial_memory)
    
    def evaluate_extracted_fsc(self, agent: FatherAgent) -> EvaluationResults:
        extracted_policy = self.extract_fsc(agent.wrapper, agent.environment)
        evaluation_result = evaluate_policy_in_model(
            extracted_policy, agent.args, agent.environment, agent.tf_environment)
        return evaluation_result
        



def store_results_in_file(model_name, memory_width, agent_evaluation_result: EvaluationResults, bottlenecked_result: EvaluationResults, evaluate_fsc = False):
    i = 0
    results_home = "experiments_small"
    if evaluate_fsc:
        results_home = "experiments_small_fsc_results"
    if not os.path.exists(results_home):
        os.makedirs(results_home)
    while os.path.exists(f"{results_home}/{model_name}_bottlenecking_results_{i}.json"):
        i += 1
    with open(f"{results_home}/{model_name}_bottlenecking_results_{i}.json", "w") as f:
        f.write(f"Memory width: {memory_width}\n")
        f.write(f"Average reward: {agent_evaluation_result.returns[-1]}\n")
        f.write(
            f"Average reachability: {agent_evaluation_result.reach_probs[-1]}\n")

        f.write(
            f"Average bottlenecked reward: {bottlenecked_result.returns[-1]}\n")
        f.write(
            f"Average bottlenecked reachability: {bottlenecked_result.reach_probs[-1]}\n")


def run_experiment(prism_path, properties_path, latent_memory_width, nr_epochs=1, num_data_steps=100, num_training_steps=50, evaluate_fsc = False):
    args = init_args(prism_path=prism_path, properties_path=properties_path)
    env, tf_env = init_environment(args)
    agent = Recurrent_PPO_agent(env, tf_env, args)
    agent.train_agent(num_training_steps)
    split_path = prism_path.split("/")
    model_name = split_path[-2]
    for size in range(2, latent_memory_width + 1): 
        if evaluate_fsc:
            for eval_fsc in [True, False]:
                extractor = BottleneckExtractor(tf_env, 64, size)
                extractor.train_autoencoder(agent.wrapper, nr_epochs, num_data_steps)
                if eval_fsc:
                    evaluation_result = extractor.evaluate_extracted_fsc(agent)
                else:
                    evaluation_result = extractor.evaluate_bottlenecking(agent)
                # evaluation_result = extractor.evaluate_extracted_fsc(agent)
                store_results_in_file(model_name, size,
                                    agent.evaluation_result, evaluation_result, eval_fsc)
        else:
            bottlenecked_actor = BottleneckedActor(agent.agent.actor_net, extractor.autoencoder)
            bottlenecked_ppo = Recurrent_PPO_agent(env, tf_env, args, actor_net=bottlenecked_actor, critic_net=agent.agent._value_net)
            bottlenecked_ppo.train_agent(num_training_steps)
            bottlenecked_ppo.evaluate_agent(vectorized = True, max_steps = bottlenecked_ppo.args.max_steps * 2)
            save_statistics_to_new_json("experiments_finetunning", model_name, "bottlenecked_ppo", bottlenecked_ppo.evaluation_result, args, split_iteration = num_training_steps)




if __name__ == '__main__':
    # Load the model and properties from command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "-f":
        # do other stuff
        prism_path = "../../models/refuel-10/sketch.templ"
        properties_path = "../../models/refuel-10/sketch.props"
        run_experiment(prism_path=prism_path, properties_path=properties_path,
                       latent_memory_width=3, nr_epochs=100, num_training_steps=1000,
                       evaluate_fsc = True)
        exit(0)
    
    if len(sys.argv) > 1:
        path = sys.argv[1]

        template_path = os.path.join(f"{path}", "sketch.templ")
        properties_path = os.path.join(f"{path}", "sketch.props")
        if len(sys.argv) > 2:
            latent_memory_width = int(sys.argv[2])
        else:
            latent_memory_width = 3
    else:
        path = "../../models/mba"
        template_path = f"{path}/sketch.templ"
        properties_path = f"{path}/sketch.props"
        latent_memory_width = 3
    print(
        f"Running experiment with latent memory width: {latent_memory_width}")
    print(f"Model path: {path}")
    run_experiment(prism_path=template_path, properties_path=properties_path,
                   latent_memory_width=latent_memory_width)
    exit(0)
    # prism_path = "../../models_large/network-5-10-8/sketch.templ"
    # properties_path = "../../models_large/network-5-10-8/sketch.props"
    # args = init_args(prism_path=prism_path, properties_path=properties_path)
    # env, tf_env = init_environment(args)
    # agent = Recurrent_PPO_agent(env, tf_env, args)
    # agent.train_agent(2000)
    # extractor = BottleneckExtractor(tf_env, 128, 32)
    # extractor.train_autoencoder(agent.wrapper, 100, 400)
    # extractor.evaluate_bottlenecking(agent)
