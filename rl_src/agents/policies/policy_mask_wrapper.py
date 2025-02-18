from tf_agents.policies import TFPolicy
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep

import numpy as np
import tensorflow as tf

from tf_agents.distributions import shifted_categorical
from tf_agents.trajectories import policy_step

import tensorflow_probability as tfp


from tools.encoding_methods import observation_and_action_constraint_splitter

from rl_src.agents.policies.parallel_fsc_policy import FSC

import logging

logger = logging.getLogger(__name__)

class PolicyMaskWrapper(TFPolicy):
    """Wrapper for stochastic policies that allows to use observation and action constraint splitters"""

    def __init__(self, policy: TFPolicy, observation_and_action_constraint_splitter=observation_and_action_constraint_splitter, 
                 time_step_spec=None, is_greedy : bool = False, select_rand_action_probability : float = 0.0):
        """Initializes the policy mask wrapper, which is a wrapper for stochastic policies which enables to use observation and action constraint splitters.

        Args:
            policy (TFPolicy): Policy, which should be wrapped. This policy does not use masks for action selection.
            observation_and_action_constraint_splitter (func, optional): Splits observations to pure observations and masks. 
                                                                         Defaults to observation_and_action_constraint_splitter from agents.tools.
            time_step_spec (TimeStepSpec, optional): Time Step specification with mask. Defaults to None.
            is_greedy (bool, optional): Whether the policy should be greedy or not. Defaults to False.
        """
        super(PolicyMaskWrapper, self).__init__(time_step_spec=time_step_spec,
                                                  action_spec=policy.action_spec,
                                                  policy_state_spec=policy.policy_state_spec,
                                                  info_spec=policy.info_spec,
                                                  observation_and_action_constraint_splitter=observation_and_action_constraint_splitter)
        
        self._policy = policy
        self._observation_and_action_constraint_splitter = observation_and_action_constraint_splitter
        self._time_step_spec = time_step_spec
        self._action_spec = policy.action_spec
        self._policy_state_spec = policy.policy_state_spec
        self._info_spec = policy.info_spec
        self._is_greedy = is_greedy
        self._real_distribution = self._distribution
        self._select_random_action_probability = select_rand_action_probability

    def is_greedy(self):
        return self._is_greedy
    
    def set_greedy(self, is_greedy):
        self._is_greedy = is_greedy

    @tf.function
    def _get_initial_state(self, batch_size):
        # print("Getting initial state", batch_size)
        return self._policy.get_initial_state(batch_size)    

    def _distribution(self, time_step, policy_state) -> PolicyStep:
        observation, mask = self._observation_and_action_constraint_splitter(
            time_step.observation)
        time_step = time_step._replace(observation=observation)
        distribution_result = self._policy.distribution(
            time_step, policy_state)
        
        # logits = distribution_result.action.logits
        # print(logits)
        # policy_state = distribution_result.state
        # Taken from q_policy.py from TensorFlow library
        # almost_neg_inf = tf.constant(logits.dtype.min, dtype=logits.dtype)
        # almost_neg_inf = tf.constant(-1e10, dtype=logits.dtype)
        # print(mask)
        # print(logits)
        # logits = tf.compat.v2.where(
        #     tf.cast(mask, tf.bool), logits, almost_neg_inf
        # )
        # print(logits)
        # distribution = tfp.distributions.Categorical(
        #     logits=logits
        # )
        # distribution = tf.nest.pack_sequence_as(
        #     self._action_spec, [distribution])
        return distribution_result
        return policy_step.PolicyStep(distribution, policy_state, distribution_result.info)
    
    def _get_action_masked(self, distribution, mask):
        logits = distribution.action.logits
        almost_neg_inf = tf.constant(logits.dtype.min, dtype=logits.dtype)
        logits = tf.compat.v2.where(
            tf.cast(mask, tf.bool), logits, almost_neg_inf
        )
        distribution = tfp.distributions.Categorical(
            logits=logits
        )
        action = distribution.sample()
        return action
    
    def _get_initial_state(self, batch_size):
        return self._policy.get_initial_state(batch_size)

    def _action(self, time_step, policy_state, seed) -> PolicyStep:
        # print(policy_state)
        observation, mask = self._observation_and_action_constraint_splitter(time_step.observation)
        time_step = time_step._replace(observation=observation)
        return self._policy.action(time_step, policy_state, seed)

        distribution = self._real_distribution(time_step, policy_state)

        if self._is_greedy:
            action = tf.argmax(distribution.action.logits, output_type=tf.int32, axis=-1)
        else:
            # _, mask = self._observation_and_action_constraint_splitter(time_step.observation)
            # action = self._get_action_masked(distribution, mask)
            action = distribution.action.sample()
            
        policy_step = PolicyStep(action=action, state=distribution.state, info=distribution.info)
        return policy_step
    
    def _get_action_entropy(self, time_step, policy_state):
        observation, mask = self._observation_and_action_constraint_splitter(time_step.observation)
        time_step = time_step._replace(observation=observation)
        distribution = self._real_distribution(time_step, policy_state)
        logits = tf.nn.softmax(distribution.action.logits)
        entropy = -tf.reduce_sum(logits * tf.math.log(logits), axis=-1)
        return entropy

    def _randomized_action(self, time_step, policy_state, seed):
        policy_step = self._action_original(time_step, policy_state, seed) 
        rand_number = np.random.uniform(0.0, 1.0)
        if rand_number < self._select_random_action_probability:
            rand_action = np.random.choice(self._action_spec.maximum, size=1)
            policy_step.action = tf.constant(rand_action)
        return policy_step
