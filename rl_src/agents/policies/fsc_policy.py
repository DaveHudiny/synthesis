# Includes implemenation of FSCPolicy class, which is a TFPolicy implementation of FSC used for Hard and Soft FSC oracle.
# Author: David Hudák
# Login: xhudak03
# File: fsc_policy.py

import random

from tools.encoding_methods import *
from environment import tf_py_environment
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.specs.tensor_spec import TensorSpec
from tf_agents.utils import common

from tf_agents.trajectories import StepType

import tensorflow_probability as tfp

import tensorflow as tf

from agents.policies.fsc_copy import FSC


import logging
logger = logging.getLogger(__name__)

class FSC_Policy(TFPolicy):
    def __init__(self, tf_environment: tf_py_environment.TFPyEnvironment, fsc: FSC,
                 observation_and_action_constraint_splitter=None, tf_action_keywords=[],
                 info_spec=None, parallel_policy : TFPolicy = None, soft_decision = False,
                 soft_decision_multiplier : float = 2.0, need_logits : bool = True,
                 switch_probability : float = None, duplex_buffering : bool = False,
                 info_mem_node = False):
        """Implementation of FSC policy based on FSC object obtained from Paynt (or elsewhere).

        Args:
            tf_environment (tf_py_environment.TFPyEnvironment): TensorFlow environment. Used for time_step_spec and action_spec.
            fsc (FSC): FSC object (usually obtained from Paynt).
            observation_and_action_constraint_splitter (func, optional): Splits observations to pure observations and masks. Defaults to None.
            tf_action_keywords (list, optional): List of action keywords. Should be in order of actions, which the tf_environment work with. Defaults to [].
            info_spec (tuple, optional): Information specification. Defaults to None.
            parallel_policy (TFPolicy, optional): Parallel stochastic policy, which generates logits. Defaults to None.
            soft_decision (bool, optional): If True, the policy will use the parallel policy to make a decision combined with its FSC. Defaults to False.
        """
        if duplex_buffering:
            self.init_duplex_buffering(info_spec)
        else:
            self._info_spec = info_spec
            self.duplex_buffering = False

        self.info_mem_node = info_mem_node

        super(FSC_Policy, self).__init__(tf_environment._time_step_spec, tf_environment._action_spec,
                                         policy_state_spec=TensorSpec(
                                             shape=(), dtype=tf.int32),
                                         observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
                                         info_spec=self._info_spec,
                                        )
        self._time_step_spec = tf_environment._time_step_spec
        self._action_spec = tf_environment._action_spec
        self._observation_and_action_constraint_splitter = observation_and_action_constraint_splitter
        self._fsc = fsc
        self._fsc.action_function = tf.constant(
            self._fsc.action_function, dtype=tf.int32)
        self._fsc.update_function = tf.constant(
            self._fsc.update_function, dtype=tf.int32)
        self._fsc_action_labels = tf.constant(
            self._fsc.action_labels, dtype=tf.string)
        self.tf_action_labels = tf.constant(tf_action_keywords, dtype=tf.string)
        self._parallel_policy = parallel_policy
        if parallel_policy is not None:
            self._parallel_policy_function = common.function(parallel_policy.action)
            self._hidden_ppo_state = self._parallel_policy.get_initial_state(tf_environment.batch_size)
        self._soft_decision = soft_decision
        self._fsc_update_coef = soft_decision_multiplier
        self.switched = False
        if switch_probability is not None:
            self.switching = True
            self.switch_probability = switch_probability
        else:
            self.switching = False

    def init_duplex_buffering(self, original_info_spec):
        self.duplex_buffering = True
        self._info_spec = {
                'fsc': TensorSpec(shape=(1,), dtype=tf.bool, name='fsc'),
                'rl': original_info_spec,
                'mem_node': TensorSpec(shape=(1,), dtype=tf.int32, name='mem_node')
        }

    tf.function
    def _set_hidden_ppo_state(self, batch_size=1):
        if self._parallel_policy is not None:
            self._hidden_ppo_state = self._parallel_policy.get_initial_state(batch_size)

    def _get_initial_state(self, batch_size):
        self._set_hidden_ppo_state(batch_size=batch_size)
        self.switched = False
        return tf.zeros((batch_size, 1), dtype=tf.int32)

    def _distribution(self, time_step, policy_state, seed):
        raise NotImplementedError(
            "FSC policy does not support distribution-based action selection")

    @tf.function
    def convert_to_tf_action_number(self, action_numbers):
        def map_action_number(action_number):
            keyword = self._fsc_action_labels[action_number]
            if keyword == "__no_label__":
                return tf.constant(-1, dtype=tf.int32)
            tf_action_number = tf.argmax(
                tf.cast(tf.equal(self.tf_action_labels, keyword), tf.int32), output_type=tf.int32)
            return tf_action_number

        tf_action_numbers = tf.map_fn(map_action_number, action_numbers, dtype=tf.int32)
        return tf_action_numbers
    
    def _make_soft_decision(self, fsc_action_number, time_step, seed):
        distribution = self._parallel_policy.distribution(time_step, self._hidden_ppo_state)
        self._hidden_ppo_state = distribution.state
        policy_info = distribution.info
        logits = distribution.action.logits
        one_hot_encoding = tf.one_hot(fsc_action_number, len(self.tf_action_labels)) * self._fsc_update_coef
        updated_logits = logits + one_hot_encoding
        action_number = tfp.distributions.Categorical(logits=updated_logits).sample()[0]
        return action_number, policy_info

    def _generate_paynt_decision(self, time_step, policy_state, seed):
        observation = time_step.observation["integer"]
        int_policy_state = tf.squeeze(policy_state)
        observation = tf.squeeze(observation)
        indices = tf.stack([int_policy_state, observation], axis=1)
        action_number = tf.gather_nd(self._fsc.action_function, indices)
        action_number = self.convert_to_tf_action_number(action_number)
        new_policy_state = tf.gather_nd(self._fsc.update_function, indices)
        new_policy_state = tf.convert_to_tensor(tf.reshape(new_policy_state, shape=(-1, 1)), dtype=tf.int32)
        return action_number, new_policy_state

    def _create_one_hot_fake_info(self, action_number):
        one_hot_encoding = tf.one_hot(action_number, len(self.tf_action_labels)) / 2.0
        one_hot_encoding_with_alternative = tf.where(one_hot_encoding == 0.0, -1.0, one_hot_encoding)
        one_hot_encoding = tf.cast(one_hot_encoding_with_alternative, tf.float32, name="CategoricalProjectionNetwork_logits")
        return {"dist_params": {"logits": one_hot_encoding}}
    
    def _get_switched_ppo_action(self, time_step, seed):
        parallel_policy_step = self._parallel_policy_function(time_step, self._hidden_ppo_state, seed)
        self._hidden_ppo_state = parallel_policy_step.state
        return parallel_policy_step.action, parallel_policy_step.info
    
    # @tf.function
    def _action(self, time_step, policy_state, seed):
        # Change the policy_state for the first time steps
        equal_mask = tf.math.equal(time_step.step_type, StepType.FIRST)
        equal_mask = tf.reshape(equal_mask, shape=(-1, 1))
        policy_state = tf.where(
            equal_mask,
            self._get_initial_state(tf.shape(time_step.observation["integer"])[0]),
            policy_state
        )
        init_state = self._parallel_policy.get_initial_state(tf.shape(time_step.observation["integer"])[0])
        for key in self._hidden_ppo_state.keys():
            for i in range(len(self._hidden_ppo_state[key])):
                self._hidden_ppo_state[key][i] = tf.where(
                    equal_mask,
                    init_state[key][i],
                    self._hidden_ppo_state[key][i]
                )

        batch_size = tf.shape(time_step.observation["integer"])[0]
        policy_info = ()
        if self.switching: # Switching between FSC and PPO
            if self.switched:
                new_policy_state = policy_state
                action_number, policy_info = self._get_switched_ppo_action(time_step, seed)
            else:
                if random.random() < self.switch_probability:
                    self.switched = True
                action_number, new_policy_state = self._generate_paynt_decision(time_step, policy_state, seed)
                policy_info = self._create_one_hot_fake_info(action_number)
        else:
            action_number, new_policy_state = self._generate_paynt_decision(time_step, policy_state, seed)
        if self._info_spec is None or self._info_spec == ():
            policy_info = ()
        
        elif self._parallel_policy is not None and policy_info == (): # Generate logits from PPO policy
            if self._soft_decision: # Use PPO policy to make a decision combined with FSC
                action_number, policy_info = self._make_soft_decision(action_number, time_step, seed)
            else: # Hard FSC decision
                parallel_policy_step = self._parallel_policy_function(time_step, self._hidden_ppo_state, seed)
                self._hidden_ppo_state = parallel_policy_step.state
                policy_info = parallel_policy_step.info
        if self.duplex_buffering:
            policy_info = {
                "fsc": tf.constant([not self.switched] * batch_size, dtype=tf.bool),
                "rl": policy_info,
                "mem_node": new_policy_state
            }
        if policy_info == () and self.info_mem_node:
            policy_info = {"mem_node" : new_policy_state}
        elif policy_info == () and self._info_spec != (): # If parallel policy does not return logits, use one-hot encoding of action number
            policy_info = self._create_one_hot_fake_info(action_number)
        policy_step = PolicyStep(action=action_number, state=new_policy_state, info=policy_info)
        return policy_step
    
