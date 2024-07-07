# Description: This file contains the implementation of the FatherAgent class, which is the parent class of other agents in the project.
# Author: David Hudák
# Login: xhudak03
# Project: diploma-thesis
# File: father_agent.py

from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from keras.optimizers import Adam

import tensorflow as tf
import tf_agents

from environment.environment_wrapper import Environment_Wrapper
from rl_src.agents.encoding_methods import *
from agents.abstract_agent import AbstractAgent
from agents.evaluators import *
from agents.policies.random_policy import Random_Policy
from agents.policies.fsc_policy import FSC_Policy, FSC

import logging

logger = logging.getLogger(__name__)

from tf_agents.policies import py_tf_eager_policy

import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class AgentSettings:
    """Class for storing information about agents. Possible usage with extension of the project."""
    def __init__(self, preprocessing_layers=[150, 150], lstm_units=[100], postprocessing_layers=[]):
        self.preprocessing_layers = preprocessing_layers
        self.lstm_units = lstm_units
        self.postprocessing_layers = postprocessing_layers


class FatherAgent(AbstractAgent):
    """Class for the parent agent of all agents in the project."""
    def load_fsc(self, fsc_json_path):
        """Load FSC from JSON file.
        
        Args:
            fsc_json_path: Path to the JSON file with FSC.
        """
        with open(fsc_json_path, 'r') as f:
            fsc_json = json.load(f)
        fsc = FSC.from_json(fsc_json)
        return fsc

    def common_init(self, environment: Environment_Wrapper, tf_environment: tf_py_environment.TFPyEnvironment, args, load=False, agent_folder=None):
        """Common initialization of the agents.

        Args:
            environment: The environment wrapper object, used for additional information about the environment.
            tf_environment: The TensorFlow environment object, used for simulation information.
            args: The arguments object for all the important settings.
            load: Whether to load the agent. Unused.
            agent_folder: The folder where the agent is stored.
        """
        self.environment = environment
        self.tf_environment = tf_environment
        self.args = args
        self.evaluation_episodes = args.evaluation_episodes
        self.agent_folder = agent_folder
        self.traj_num_steps = args.num_steps
        self.agent = None
        if args.action_filtering:
            self.observation_and_action_constraint_splitter = None
        else:
            self.observation_and_action_constraint_splitter = observation_and_action_constraint_splitter
        if args.paynt_fsc_imitation:
            self.fsc = self.load_fsc(args.paynt_fsc_json)
        self.wrapper = None
        self.evaluation_result = EvaluationResults(environment.goal_value)

    def __init__(self, environment: Environment_Wrapper, tf_environment: tf_py_environment.TFPyEnvironment, args, load=False, agent_folder=None):
        """Initialization of the father agent. Not recommended to use this class directly, use the child classes instead. Implemented as example.
        
        Args:
            environment: The environment wrapper object, used for additional information about the environment.
            tf_environment: The TensorFlow environment object, used for simulation information.
            args: The arguments object for all the important settings.
            load: Whether to load the agent. Unused.
            agent_folder: The folder where the agent is stored."""

        self.common_init(self, environment, tf_environment,
                         args, load, agent_folder)
        train_step_counter = tf.Variable(0)
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)

        self.fc_layer_params = (10,)
        dense_layers = [tf.keras.layers.Dense(
            num_units, activation='relu') for num_units in self.fc_layer_params]

        q_values_layer = tf.keras.layers.Dense(
            units=len(environment.action_keywords),
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2)
        )

        q_net = sequential.Sequential(dense_layers + [q_values_layer])
        lstm = tf.keras.layers.LSTM(
            10, return_sequences=True, return_state=True, activation='tanh', dtype=tf.float32)
        q_net = sequential.Sequential([lstm, q_net])

        self.agent = dqn_agent.DqnAgent(
            tf_environment._time_step_spec,
            tf_environment._action_spec,
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter,
            observation_and_action_constraint_splitter=self.observation_and_action_constraint_splitter
        )
        self.agent.initialize()
        self.init_replay_buffer(tf_environment)
        self.init_collector_driver(tf_environment)

    def init_replay_buffer(self, tf_environment):
        """Initialize the uniform replay buffer for the agent.
        
        Args:
            tf_environment: The TensorFlow environment object, used for providing important specifications.
        """
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=tf_environment.batch_size,
            max_length=self.args.buffer_size)

    def init_collector_driver(self, tf_environment):
        """Initialize the collector driver for the agent.

        Args:
            tf_environment: The TensorFlow environment object, used for simulation information.
        """
        eager = py_tf_eager_policy.PyTFEagerPolicy(
            self.agent.collect_policy, use_tf_function=True, batch_time_steps=False)
        self.driver = tf_agents.drivers.dynamic_step_driver.DynamicStepDriver(
            tf_environment,
            eager,
            observers=[self.replay_buffer.add_batch],
            num_steps=self.traj_num_steps)

    def init_random_collector_driver(self, tf_environment: tf_py_environment.TFPyEnvironment):
        """Initialize the random policy collector driver for the agent. Used for random exploration.

        Args:
            tf_environment: The TensorFlow environment object, used for simulation information.
        """
        random_policy = tf_agents.policies.random_tf_policy.RandomTFPolicy(tf_environment.time_step_spec(),
                                                                           tf_environment.action_spec(),
                                                                           observation_and_action_constraint_splitter=self.observation_and_action_constraint_splitter)
        self.random_driver = tf_agents.drivers.dynamic_step_driver.DynamicStepDriver(
            tf_environment,
            random_policy,
            observers=[self.replay_buffer.add_batch],
            num_steps=self.traj_num_steps
        )

    def get_initial_state(self, batch_size=None):
        """Get the initial state of the agent."""
        return self.agent.policy.get_initial_state(batch_size=batch_size)

    def init_fsc_policy_driver(self, tf_environment: tf_py_environment.TFPyEnvironment, fsc : FSC = None, soft_decision=False, fsc_multiplier=2.0):
        """Initialize the FSC policy driver for the agent. Used for imitation learning with FSC.

        Args:
            tf_environment: The TensorFlow environment object, used for simulation information.
            fsc: The FSC object for imitation learning.
            soft_decision: Whether to use soft decision for FSC. Used only in PPO initialization.
            fsc_multiplier: The multiplier for the FSC. Used only in PPO initialization.
        """
        self.fsc_policy = FSC_Policy(tf_environment, fsc,
                                     observation_and_action_constraint_splitter=self.observation_and_action_constraint_splitter,
                                     tf_action_keywords=self.environment.action_keywords,
                                     info_spec=self.agent.policy.info_spec)
        eager = py_tf_eager_policy.PyTFEagerPolicy(
            self.fsc_policy, use_tf_function=True, batch_time_steps=False)
        
        self.fsc_driver = tf_agents.drivers.dynamic_episode_driver.DynamicEpisodeDriver(
            tf_environment,
            eager,
            observers=[self.replay_buffer.add_batch],
            num_episodes=1
        )

    def get_evaluated_policy(self):
        """Get the policy for evaluation. Important, when using wrappers."""
        if self.wrapper is None:
            return self.agent.policy
        else:
            return self.wrapper

    def train_agent_off_policy(self, num_iterations):
        """Train the agent off-policy. Main training function for the agents.

        Args:
            num_iterations: The number of iterations for training.
        """
        if self.args.paynt_fsc_imitation:
            self.init_fsc_policy_driver(self.tf_environment, self.fsc)
        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3, sample_batch_size=self.args.batch_size, num_steps=self.traj_num_steps, single_deterministic_pass=False).prefetch(3)
        self.iterator = iter(self.dataset)
        logger.info("Training agent")
        self.agent.train = common.function(self.agent.train)
        if self.agent.train_step_counter.numpy() == 0:
            logger.info('Random Average Return = {0}'.format(compute_average_return(
                self.get_evaluated_policy(), self.tf_environment, self.evaluation_episodes, self.environment)))
        for _ in range(5): # Because sometimes FSC driver does not sample enough trajectories to start learning.
            self.driver.run()
        for i in range(num_iterations):
            if False:
                self.random_driver.run()
            if (self.args.paynt_fsc_imitation or hasattr(self, "fsc_driver")) and i < self.args.fsc_policy_max_iteration:
                self.fsc_driver.run()
            else:
                self.driver.run()
            experience, _ = next(self.iterator)
            train_loss = self.agent.train(experience).loss
            train_loss = train_loss.numpy()
            self.agent.train_step_counter.assign_add(1)
            self.evaluation_result.add_loss(train_loss)
            if i % 10 == 0:
                logger.info(f"Step: {i}, Training loss: {train_loss}")
            if i % 100 == 0:
                self.evaluate_agent()
        self.evaluate_agent(True)
                
        self.replay_buffer.clear()

    def evaluate_agent(self, last=False):
        """Evaluate the agent. Used for evaluation of the agent during training.

        Args:
            last: Whether this is the last evaluation of the agent.
        """
        if self.args.prefer_stochastic:
            self.set_agent_stochastic()
        else:
            self.set_agent_greedy()
        if last:
            evaluation_episodes = self.evaluation_episodes * 2
        else:
            evaluation_episodes = self.evaluation_episodes
        compute_average_return(
                self.get_evaluated_policy(), self.tf_environment, evaluation_episodes, self.environment, self.evaluation_result.update)
        
        self.set_agent_stochastic()
        if self.evaluation_result.best_updated:
            self.save_agent(best=True)
        logger.info('Average Return = {0}'.format(self.evaluation_result.returns[-1]))
        logger.info('Average Virtual Goal Value = {0}'.format(self.evaluation_result.returns_episodic[-1]))
        logger.info('Goal Reach Probability = {0}'.format(self.evaluation_result.reach_probs[-1]))
    
    def set_agent_greedy(self):
        """Set the agent for to be greedy for evaluation. Used only with PPO agent, where we select greedy evaluation.
        """
        if self.wrapper is None:
            pass
        else:
            self.wrapper.set_greedy(True)

    def set_agent_stochastic(self):
        """Set the agent to be stochastic for evaluation. Used only with PPO agent, where we select stochastic evaluation.
        """
        if self.wrapper is None:
            pass
        else:
            self.wrapper.set_greedy(False)

    def policy(self, time_step, policy_state=None):
        """Make a decision based on the policy of the agent."""
        if policy_state is None:
            policy_state = self.agent.policy.get_initial_state(None)
        return self.agent.policy.action(time_step, policy_state=policy_state)

    def save_agent(self, best=False):
        """Save the agent. Used for saving the agent after or during training training.
        
        Args:
            best: Whether this is the best agent. If true, the agent is saved in the best folder.
        """
        if self.agent is None or tf.train is None:
            logger.info("No agent for saving.")
            return
        checkpoint = tf.train.Checkpoint(agent=self.agent)
        if best:
            agent_folder = self.agent_folder + "/best"
        else:
            agent_folder = self.agent_folder
        manager = tf.train.CheckpointManager(
            checkpoint, agent_folder, max_to_keep=5)
        manager.save()

    def load_agent(self, best=False):
        """Load the agent. Used for loading the agent after or during training.

        Args:
            best: Whether this is the best agent. If true, the agent is loaded from the best folder.
        """
        checkpoint = tf.train.Checkpoint(agent=self.agent)
        if best:
            agent_folder = self.agent_folder + "/best"
        else:
            agent_folder = self.agent_folder
        manager = tf.train.CheckpointManager(
            checkpoint, agent_folder, max_to_keep=5)
        latest_checkpoint = manager.latest_checkpoint
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            logger.info(f"Loaded data from checkpoint: {latest_checkpoint}")
        else:
            logger.info("No data for loading.")

    def reset_weights(self):
        """Reset the weights of the agent. Implemented in the child classes."""
        raise NotImplementedError
