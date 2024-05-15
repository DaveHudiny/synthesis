from tf_agents.policies import TFPolicy
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep

import numpy as np
import tensorflow as tf

from tf_agents.distributions import shifted_categorical
from tf_agents.trajectories import policy_step

import tensorflow_probability as tfp


from rl_src.agents.encoding_methods import observation_and_action_constraint_splitter

from agents.policies.fsc_policy import FSC

import logging

logger = logging.getLogger(__name__)

class Policy_Mask_Wrapper(TFPolicy):
    """Wrapper for stochastic policies that allows to use observation and action constraint splitters"""

    def __init__(self, policy: TFPolicy, observation_and_action_constraint_splitter=observation_and_action_constraint_splitter, 
                 time_step_spec=None, is_greedy=False):
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
        self._fsc_memory = tf.constant([0], dtype=tf.int32)
        self._real_distribution = self._distribution

    def is_greedy(self):
        return self._is_greedy
    
    def set_greedy(self, is_greedy):
        self._is_greedy = is_greedy

    def _prepare_fsc_oracle(self, number_of_possible_observations : int): # Unused in final version
        self._fsc_action_function = tf.zeros(
            [self._action_spec.maximum + 2, number_of_possible_observations], dtype=tf.int32)
        self._fsc_update_function = tf.zeros(
            [self._action_spec.maximum + 2, number_of_possible_observations], dtype=tf.int32)
        labels_list = ["__no_label__" for _ in range(self._action_spec.maximum + 2)]
        # +2 because of __no_label__ actions in the used FSC.
        self._fsc_action_labels = tf.constant(labels_list, dtype=tf.string)
        self.tf_action_labels = tf.constant(labels_list, dtype=tf.string)

    tf.function
    def _get_initial_state(self, batch_size):
        if self._fsc_pre_init is not None:
            self._fsc_memory = tf.constant([0], dtype=tf.int32)
        return self._policy.get_initial_state(batch_size)
    
    def _update_fsc_oracle(self, fsc_oracle : FSC, tf_action_keywords = None): # Unused in final version
        print("Shape of action function: ", self._fsc_action_function)
        print("Shape of update function: ", self._fsc_update_function)
        print("Shape of action labels: ", self._fsc_action_labels)
        print("Shape of tf action labels: ", self.tf_action_labels)
        
        print("Shape of fsc_oracle action function: ", np.array(fsc_oracle.action_function).shape)
        print("Shape of fsc_oracle update function: ", np.array(fsc_oracle.update_function).shape)
        print("Shape of fsc_oracle action labels: ", np.array(fsc_oracle.action_labels).shape)
        print("Shape of tf action keywords: ", np.array(tf_action_keywords).shape)

        update_indices = tf.where(tf.not_equal(fsc_oracle.action_function, -1))
        print(update_indices)
        print(tf.constant(fsc_oracle.action_function)[update_indices[0, 0].numpy(), update_indices[0, 1].numpy()])
        self._fsc_action_function = tf.tensor_scatter_nd_update(self._fsc_action_function,
                                                                update_indices,
                                                                tf.constant(fsc_oracle.action_function))

        update_indices = tf.where(tf.not_equal(fsc_oracle.update_function, -1))
        self._fsc_update_function = tf.tensor_scatter_nd_update(self._fsc_update_function,
                                                                update_indices,
                                                                tf.constant(fsc_oracle.update_function))
        update_indices = tf.where(tf.not_equal(fsc_oracle.action_labels, ""))
        self._fsc_action_labels = tf.tensor_scatter_nd_update(self._fsc_action_labels,
                                                              update_indices,
                                                              tf.constant(fsc_oracle.action_labels))
        update_indices = tf.where(tf.not_equal(tf_action_keywords, ""))
        self.tf_action_labels = tf.tensor_scatter_nd_update(self.tf_action_labels,
                                                            update_indices,
                                                            tf.constant(tf_action_keywords))

    def _set_fsc_oracle(self, fsc_oracle : FSC, tf_action_keywords = None): # Unused in final version
        
        self._fsc_action_function = tf.constant(
            fsc_oracle.action_function, dtype=tf.int32)
        self._fsc_update_function = tf.constant(
            fsc_oracle.update_function, dtype=tf.int32)
        self._fsc_action_labels = tf.constant(
            fsc_oracle.action_labels, dtype=tf.string)
        self.tf_action_labels = tf.constant(tf_action_keywords, dtype=tf.string)
        self._real_distribution = self._fsc_biased_distribution

        

    def _unset_fsc_oracle(self): # Unused in final version
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

    @tf.function
    def convert_to_tf_action_number(self, action_number):
        keyword = self._fsc_action_labels[action_number]
        if keyword == "__no_label__":
            return tf.constant(-1, dtype=tf.int32)
        tf_action_number = tf.argmax(
            tf.cast(tf.equal(self.tf_action_labels, keyword), tf.int32), output_type=tf.int32)
        return tf_action_number
    
    @tf.function
    def _fsc_biased_distribution(self, time_step, policy_state):
        fsc_observation = time_step.observation["integer"]
        int_policy_state = tf.squeeze(self._fsc_memory)
        fsc_observation = tf.squeeze(fsc_observation)
        action_number = self._fsc_action_function[int_policy_state][fsc_observation]
        action_number = self.convert_to_tf_action_number(action_number)
        self._fsc_memory = self._fsc_update_function[int_policy_state][fsc_observation]

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
        logits = logits[action_number] + 0.3
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
