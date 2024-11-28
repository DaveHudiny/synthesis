# Implementation of PPO with critic represetned by Q-values obtained from FSC.
# Author: David Hudák
# Login: xhudak03
# File: ppo_with_qvalues_fsc.py

from agents.father_agent import FatherAgent
from environment.environment_wrapper import Environment_Wrapper

from agents.networks.value_networks import get_alternative_call_of_qnet, create_recurrent_value_net_demasked, Value_DQNet
from agents.networks.actor_networks import create_recurrent_actor_net_demasked
from agents.policies.policy_mask_wrapper import Policy_Mask_Wrapper

from tools.encoding_methods import observation_and_action_constraint_splitter


from environment import tf_py_environment
from tf_agents.agents.ppo import ppo_agent
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.policies import py_tf_eager_policy
from tf_agents.networks.value_rnn_network import ValueRnnNetwork
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.utils import common

import tensorflow as tf
import tf_agents

import logging

from paynt.rl_extension.saynt_rl_tools.behavioral_trainers import Actor_Value_Pretrainer
from paynt.quotient.fsc import FSC

logger = logging.getLogger(__name__)


class PPO_with_External_Networks(FatherAgent):
    def __init__(self, environment: Environment_Wrapper, tf_environment: tf_py_environment.TFPyEnvironment,
                 args, load=False, agent_folder=None, dqn_agent: DqnAgent = None,
                 actor_net: ActorDistributionRnnNetwork = None, critic_net: ValueRnnNetwork = None):

        self.common_init(environment, tf_environment, args, load, agent_folder)
        tf_environment = self.tf_environment
        self.agent = None
        self.policy_state = None

        if actor_net is not None:
            self.actor_net = actor_net
        else:
            self.actor_net = create_recurrent_actor_net_demasked(
                tf_environment, tf_environment.action_spec())

        if critic_net is not None:
            self.critic_net = critic_net
        elif dqn_agent is not None:
            self.dqn_critic_net = dqn_agent._q_network
            self.critic_net = Value_DQNet(self.dqn_critic_net)
        else:
            raise ValueError(
                "This type of network expects at least one external critic.")

        time_step_spec = tf_environment.time_step_spec()
        time_step_spec = time_step_spec._replace(
            observation=tf_environment.observation_spec()["observation"])

        self.agent = ppo_agent.PPOAgent(
            time_step_spec,
            tf_environment.action_spec(),
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=args.learning_rate),
            actor_net=self.actor_net,
            value_net=self.critic_net,
            num_epochs=4,
            train_step_counter=tf.Variable(0),
            greedy_eval=False,
            discount_factor=0.99,
            use_gae=True,
            # lambda_value=0.82,
            # gradient_clipping=0.9,
            # policy_l2_reg=0.00001,
            # value_function_l2_reg=0.00001,
            # value_pred_loss_coef=0.4,
        )

        # self.agent = ppo_agent.PPOAgent(
        #     time_step_spec,
        #     tf_environment.action_spec(),
        #     actor_net=self.actor_net,
        #     value_net=self.critic_net,
        #     optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        #     normalize_observations=True,
        #     normalize_rewards=True,
        #     entropy_regularization=0.01,
        #     use_gae=True,
        #     num_epochs=3,
        #     debug_summaries=False,
        #     summarize_grads_and_vars=False,
        #     train_step_counter=tf.Variable(0),
        #     lambda_value=0.85,
        #     name='PPO_with_Pretraining',
        #     greedy_eval=False,
        #     discount_factor=0.95
        # )
        self.agent.initialize()
        logging.info("Agent initialized")

        self.args.prefer_stochastic = True
        self.init_replay_buffer()
        logging.info("Replay buffer initialized")
        self.init_collector_driver(
            self.tf_environment, demasked=True, alternative_observer=None)
        self.wrapper = Policy_Mask_Wrapper(self.agent.policy, observation_and_action_constraint_splitter, tf_environment.time_step_spec(),
                                           is_greedy=False)
        # self.wrapper = self.agent.policy
        if load:
            self.load_agent()

    def train_iteration_rl_fsc(self, pre_trainer: Actor_Value_Pretrainer, experience_rl, experience_fsc, critic_only: bool = False):
        train_loss = self.agent.train(experience_rl).loss
        if critic_only:
            actor_loss = pre_trainer.train_actor_iteration(
                actor_net=self.agent._actor_net, experience=experience_fsc)
        else:
            actor_loss = ()
        critic_loss = pre_trainer.train_value_iteration(
            critic_net=self.agent._value_net, experience=experience_fsc)
        return train_loss, actor_loss, critic_loss

    def train_duplex(self, epochs: int, fsc: FSC, pre_trainer: Actor_Value_Pretrainer, critic_only: bool = False):
        duplex_driver = pre_trainer.get_duplex_driver(fsc=fsc, rl_agent=self.agent,
                                                      replay_buffer_fsc=pre_trainer.replay_buffer,
                                                      replay_buffer_rl=self.replay_buffer,
                                                      parallel_policy=self.wrapper)

        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=4, sample_batch_size=self.args.batch_size, num_steps=self.traj_num_steps,
            single_deterministic_pass=False
        ).prefetch(4)

        self.dataset_fsc = pre_trainer.replay_buffer.as_dataset(
            num_parallel_calls=4, sample_batch_size=self.args.batch_size, num_steps=self.traj_num_steps,
            single_deterministic_pass=False
        ).prefetch(4)

        iterator = iter(self.dataset)
        iterator_fsc = iter(self.dataset_fsc)

        logger.info("Training agent")
        pre_trainer.reinit_fsc_policy_driver(fsc)
        for _ in range(7):
            duplex_driver.run()
            self.driver.run()
            pre_trainer.fill_replay_buffer_with_fsc()
        self.agent.train = common.function(self.agent.train)
        for i in range(epochs):
            self.driver.run()
            pre_trainer.fill_replay_buffer_with_fsc()
            experience_rl, _ = next(iterator)
            experience_fsc, _ = next(iterator_fsc)
            train_loss, actor_loss, critic_loss = self.train_iteration_rl_fsc(pre_trainer, experience_rl=experience_rl,
                                                                              experience_fsc=experience_fsc, critic_only=critic_only)
            train_loss = train_loss.numpy()
            self.agent.train_step_counter.assign_add(1)
            self.evaluation_result.add_loss(train_loss)
            if i % 10 == 0:
                logger.info(f"Step: {i}, Training loss: {train_loss}")
            if i % 100 == 0:
                self.environment.set_random_starts_simulation(False)
                self.evaluate_agent()
        self.environment.set_random_starts_simulation(False)
        self.evaluate_agent(last=True)
        self.replay_buffer.clear()

    def init_collector_driver_ppo(self, tf_environment: tf_py_environment.TFPyEnvironment):
        self.collect_policy_wrapper = Policy_Mask_Wrapper(
            self.agent.collect_policy,
            observation_and_action_constraint_splitter,
            tf_environment.time_step_spec(),
            select_rand_action_probability=0.00
        )
        # self.collect_policy_wrapper = self.agent.collect_policy
        eager = py_tf_eager_policy.PyTFEagerPolicy(
            self.collect_policy_wrapper, use_tf_function=True, batch_time_steps=False)
        # eager = self.collect_policy_wrapper
        observer = self.get_demasked_observer()
        # observer = self.replay_buffer.add_batcyh
        self.driver = tf_agents.drivers.dynamic_step_driver.DynamicStepDriver(
            tf_environment,
            eager,
            observers=[observer],
            num_steps=self.traj_num_steps)
