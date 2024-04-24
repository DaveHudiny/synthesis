from tf_agents.policies import TFPolicy
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep

import numpy as np
import tensorflow as tf

from tf_agents.distributions import shifted_categorical
from tf_agents.trajectories import policy_step

import tensorflow_probability as tfp


from agents.tools import observation_and_action_constraint_splitter

from agents.policies.fsc_policy import FSC_Policy


class Policy_Mask_Wrapper(TFPolicy):
    """Wrapper for stochastic policies that allows to use observation and action constraint splitters"""

    def __init__(self, policy: TFPolicy, observation_and_action_constraint_splitter=observation_and_action_constraint_splitter, 
                 time_step_spec=None, is_greedy=False, fsc_oracle : TFPolicy = None):
        """Initializes the policy mask wrapper, which is a wrapper for stochastic policies which enables to use observation and action constraint splitters.

        Args:
            policy (TFPolicy): Policy, which should be wrapped. This policy does not use masks for action selection.
            observation_and_action_constraint_splitter (func, optional): Splits observations to pure observations and masks. 
                                                                         Defaults to observation_and_action_constraint_splitter from agents.tools.
            time_step_spec (TimeStepSpec, optional): Time Step specification with mask. Defaults to None.
            is_greedy (bool, optional): Whether the policy should be greedy or not. Defaults to False.
        """
        super(Policy_Mask_Wrapper, self).__init__(time_step_spec=time_step_spec,
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
        self._fsc_oracle = fsc_oracle
        if fsc_oracle is None:
            self._real_distribution = self._distribution
        else:
            self._set_fsc_oracle(fsc_oracle)

    def is_greedy(self):
        return self._is_greedy
    
    def set_greedy(self, is_greedy):
        self._is_greedy = is_greedy

    tf.function
    def _get_initial_state(self, batch_size):
        if self._fsc_oracle is not None:
            self._fsc_memory = self._fsc_oracle.get_initial_state(batch_size)
        return self._policy.get_initial_state(batch_size)
    
    def _set_fsc_oracle(self, fsc_oracle : TFPolicy):
        self._fsc_memory = fsc_oracle.get_initial_state(1)
        self._fsc_oracle = fsc_oracle
        self._real_distribution = self._fsc_biased_distribution

    def _unset_fsc_oracle(self):
        self._real_distribution = self._distribution

    def _distribution(self, time_step, policy_state):
        observation, mask = self._observation_and_action_constraint_splitter(
            time_step.observation)
        time_step = time_step._replace(observation=observation)
        distribution_result = self._policy.distribution(
            time_step, policy_state)
        logits = distribution_result.action.logits
        policy_state = distribution_result.state
        # Taken from q_policy.py from TensorFlow library
        almost_neg_inf = tf.constant(logits.dtype.min, dtype=logits.dtype)
        logits = tf.compat.v2.where(
            tf.cast(mask, tf.bool), logits, almost_neg_inf
        )
        distribution = tfp.distributions.Categorical(
            logits=logits
        )
        distribution = tf.nest.pack_sequence_as(
            self._action_spec, [distribution])
        return policy_step.PolicyStep(distribution, policy_state, distribution_result.info)

    def _fsc_biased_distribution(self, time_step, policy_state):
        observation, mask = self._observation_and_action_constraint_splitter(
            time_step.observation)
        time_step = time_step._replace(observation=observation)
        distribution_result = self._policy.distribution(
            time_step, policy_state)
        fsc_step = self._fsc_oracle.action(time_step, self._fsc_memory, seed=None)
        logits = distribution_result.action.logits
        policy_state = distribution_result.state
        self._fsc_memory = fsc_step.state
        # Taken from q_policy.py from TensorFlow library
        almost_neg_inf = tf.constant(logits.dtype.min, dtype=logits.dtype)
        logits = tf.compat.v2.where(
            tf.cast(mask, tf.bool), logits, almost_neg_inf
        )
        logits = logits[fsc_step.action] + 0.3
        distribution = tfp.distributions.Categorical(
            logits=logits
        )
        distribution = tf.nest.pack_sequence_as(
            self._action_spec, [distribution])
        return policy_step.PolicyStep(distribution, policy_state, distribution_result.info)
    
    def _action(self, time_step, policy_state, seed):
        distribution = self._real_distribution(time_step, policy_state)
        if self._is_greedy:
            action = tf.argmax(distribution.action.logits, output_type=tf.int32, axis=-1)
        else:
            action = distribution.action.sample()

        policy_step = PolicyStep(action=action, state=distribution.state, info=distribution.info)
        return policy_step
