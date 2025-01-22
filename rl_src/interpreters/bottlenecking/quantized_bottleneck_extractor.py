import tensorflow as tf
from keras import layers, models
import numpy as np

from tf_agents.policies import TFPolicy
from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

from interpreters.bottlenecking.bottleneck_autoencoder import Encoder, Decoder, Autoencoder

from environment.tf_py_environment import TFPyEnvironment
from agents.father_agent import FatherAgent

class BottlenecExtractor:
    def __init__(self, tf_environment : TFPyEnvironment):
        self.tf_environment = tf_environment
        self.autoencoder = Autoencoder(128, 32, 16)
        
        self.dataset = []
        
    def create_generator(self, policy : TFPolicy) -> iter:
        """ Creates tensorflow generator from dataset, that produces batches of data, where x and y are the same.
        """
        self.tf_dataset = tf.data.Dataset.from_tensor_slices(self.dataset).batch(64).repeat()
        return iter(zip(self.tf_dataset, self.tf_dataset))

    def collect_data(self, num_data_steps : int, policy : TFPolicy):
        eager = PyTFEagerPolicy(policy, use_tf_function=True, batch_time_steps=False)
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

    def train_autoencoder(self, policy : TFPolicy, num_epochs, num_data_steps):
        self.collect_data(num_data_steps, policy)
        generator = self.create_generator(policy)
        
        self.autoencoder.compile(optimizer='adam', loss='mse')
        self.autoencoder.fit(generator, epochs=num_epochs, steps_per_epoch=500)
        for i in range(num_epochs):
            x, y = next(generator)
            print(self.autoencoder.get_discrete_state(x))

if __name__ == '__main__':
    # Test the ExtractedFSCPolicy class on a random Policy
    from rl_src.tests.general_test_tools import *
    from rl_src.agents.recurrent_ppo_agent import Recurrent_PPO_agent
    prism_path = "../../models/mba/sketch.templ"
    properties_path = "../../models/mba/sketch.props"
    args = init_args(prism_path=prism_path, properties_path=properties_path)
    env, tf_env = init_environment(args)
    agent_policy = Recurrent_PPO_agent(env, tf_env, args)
    # agent_policy.train_agent(100)
    extractor = BottlenecExtractor(tf_env)
    extractor.train_autoencoder(agent_policy.wrapper, 10, 100)

