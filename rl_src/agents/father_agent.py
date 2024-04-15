from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from keras.optimizers import Adam

import tensorflow as tf
import tf_agents

from environment.environment_wrapper import Environment_Wrapper
from agents.tools import *
from agents.abstract_agent import AbstractAgent
from agents.evaluators import *
from agents.policies.random_policy import Random_Policy
from agents.policies.fsc_policy import FSC_Policy, FSC

import logging

logger = logging.getLogger(__name__)

from tf_agents.policies import py_tf_eager_policy


class AgentSettings:
    def __init__(self, preprocessing_layers=[150, 150], lstm_units=[100], postprocessing_layers=[]):
        self.preprocessing_layers = preprocessing_layers
        self.lstm_units = lstm_units
        self.postprocessing_layers = postprocessing_layers


class FatherAgent(AbstractAgent):
    def load_fsc(self, fsc_json_path):
        with open(fsc_json_path, 'r') as f:
            fsc_json = json.load(f)
        fsc = FSC.from_json(fsc_json)
        # action_keywords = self.environment.action_keywords
        # fsc_policy = FSC_Policy(self.tf_environment, fsc,
        #                         tf_action_keywords=action_keywords,
        #                         info_spec=self.agent.policy.info_spec)
        return fsc

    def common_init(self, environment: Environment_Wrapper, tf_environment: tf_py_environment.TFPyEnvironment, args, load=False, agent_folder=None):
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
        self.stats_without_ending = []
        self.stats_with_ending = []
        self.losses = []

    def __init__(self, environment: Environment_Wrapper, tf_environment: tf_py_environment.TFPyEnvironment, args, load=False, agent_folder=None):
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
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=tf_environment.batch_size,
            max_length=self.args.buffer_size)

    def init_collector_driver(self, tf_environment):
        eager = py_tf_eager_policy.PyTFEagerPolicy(
            self.agent.collect_policy, use_tf_function=True, batch_time_steps=False)
        self.driver = tf_agents.drivers.dynamic_step_driver.DynamicStepDriver(
            tf_environment,
            eager,
            observers=[self.replay_buffer.add_batch],
            num_steps=self.traj_num_steps)

    def init_random_collector_driver(self, tf_environment: tf_py_environment.TFPyEnvironment):
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
        return self.agent.policy.get_initial_state(batch_size=batch_size)

    def init_fsc_policy_driver(self, tf_environment: tf_py_environment.TFPyEnvironment):
        self.fsc_policy = FSC_Policy(tf_environment, self.fsc,
                                     observation_and_action_constraint_splitter=self.observation_and_action_constraint_splitter,
                                     tf_action_keywords=self.environment.action_keywords,
                                     info_spec=self.agent.policy.info_spec)
        eager = py_tf_eager_policy.PyTFEagerPolicy(
            self.fsc_policy, use_tf_function=True, batch_time_steps=False)
        self.fsc_driver = tf_agents.drivers.dynamic_step_driver.DynamicStepDriver(
            tf_environment,
            eager,
            observers=[self.replay_buffer.add_batch],
            num_steps=self.traj_num_steps
        )

    def select_evaluated_policy(self):
        if self.wrapper is None:
            return self.agent.policy
        else:
            return self.wrapper

    def train_agent(self, num_iterations, batch_size=32):
        if self.args.paynt_fsc_imitation:
            self.init_fsc_policy_driver(self.tf_environment)
        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=4, sample_batch_size=batch_size, num_steps=self.traj_num_steps, single_deterministic_pass=False).prefetch(16)
        self.iterator = iter(self.dataset)
        logger.info("Training agent")
        self.best_iteration_final = 0.0
        self.best_iteration_steps = -tf.float32.min
        self.agent.train = common.function(self.agent.train)

        logger.info('Random Average Return = {0}'.format(compute_average_return(
              self.select_evaluated_policy(), self.tf_environment, self.evaluation_episodes, self.args.using_logits)))
        for i in range(num_iterations):
            if False:
                self.random_driver.run()
            if self.args.paynt_fsc_imitation and i < self.args.fsc_policy_max_iteration:
                self.fsc_driver.run()
            else:
                self.driver.run()
            experience, _ = next(self.iterator)
            train_loss = self.agent.train(experience).loss
            train_loss = train_loss.numpy()
            self.agent.train_step_counter.assign_add(1)
            self.losses.append(train_loss)
            if i % 10 == 0:
                logger.info(f"Step: {i}, Training loss: {train_loss}")
            if i % 100 == 0:
                self.evaluate_agent()
        self.evaluate_agent(True)
                
        self.replay_buffer.clear()

    def evaluate_agent(self, last=False):
        self.set_agent_evaluation()
        if last:
            evaluation_episodes = self.evaluation_episodes * 2
        else:
            evaluation_episodes = self.evaluation_episodes
        average_return, average_episode_return = compute_average_return(
                self.select_evaluated_policy(), self.tf_environment, evaluation_episodes, self.args.using_logits)
        self.set_agent_training()
        if self.best_iteration_final < average_episode_return:
            self.best_iteration_final = average_episode_return
            self.best_iteration_steps = average_return
            self.save_agent(True)
        elif self.best_iteration_final == average_episode_return:
            if self.best_iteration_steps < average_return:
                self.best_iteration_steps = average_return
                self.save_agent(True)
        logger.info('Average Return without Virtual Goal = {0}'.format(average_return))
        logger.info('Average Virtual Goal Value = {0}'.format(average_episode_return))
        self.stats_without_ending.append(average_return)
        self.stats_with_ending.append(average_episode_return)
    
    def set_agent_evaluation(self):
        if self.wrapper is None:
            pass
        else:
            self.wrapper.set_greedy(True)

    def set_agent_training(self):
        if self.wrapper is None:
            pass
        else:
            self.wrapper.set_greedy(False)

    def policy(self, time_step, policy_state=None):
        if policy_state is None:
            policy_state = self.agent.policy.get_initial_state(None)
        return self.agent.policy.action(time_step, policy_state=policy_state)

    def compute_logit_policy(self, time_step, policy_state=None):
        action = self.agent.policy.action(time_step, policy_state=policy_state)
        logits = self.agent.policy.distribution(
            time_step, policy_state=policy_state).action.logits
        policy_stepino = tf_agents.trajectories.policy_step.PolicyStep(
            action={"action": action.action, "logits": logits},
            state=action.state,
            info=action.info,
        )
        return policy_stepino

    def save_agent(self, best=False):
        checkpoint = tf.train.Checkpoint(agent=self.agent)
        if best:
            agent_folder = self.agent_folder + "/best"
        else:
            agent_folder = self.agent_folder
        manager = tf.train.CheckpointManager(
            checkpoint, agent_folder, max_to_keep=5)
        manager.save()

    def load_agent(self, last=False):
        checkpoint = tf.train.Checkpoint(agent=self.agent)
        if last:
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
