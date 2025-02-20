# Includes implemenation of ParallelFSCPolicy class, which is a TFPolicy implementation of FSC with emulated recurrent feedback to allow trajectory learning.
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
                 info_spec=None, parallel_policy: TFPolicy = None, soft_decision=False,
                 soft_decision_multiplier: float = 2.0, need_logits: bool = True,
                 switch_probability: float = None, duplex_buffering: bool = False,
                 info_mem_node=False, independent_switches=False):
        """Implementation of FSC policy based on FSC object obtained from Paynt (or elsewhere).

        Args:
            tf_environment (tf_py_environment.TFPyEnvironment): TensorFlow environment. Used for time_step_spec and action_spec.
            fsc (FSC): FSC object (usually obtained from Paynt).
            observation_and_action_constraint_splitter (func, optional): Splits observations to pure observations and masks. Defaults to None.
            tf_action_keywords (list, optional): List of action keywords. Should be in order of actions, which the tf_environment work with. Defaults to [].
            info_spec (tuple, optional): Information specification. Defaults to None.
            parallel_policy (TFPolicy, optional): Parallel stochastic policy, which generates logits. Defaults to None.
            soft_decision (bool, optional): If True, the policy will use the parallel policy to make a decision combined with its FSC. Defaults to False.
            soft_decision_multiplier (float, optional): Multiplier for logits from parallel policy. Defaults to 2.0.
            need_logits (bool, optional): If True, the policy will generate logits from parallel policy. Defaults to True.
            switch_probability (float, optional): Probability of switching to parallel policy. Defaults to None.
            duplex_buffering (bool, optional): If True, the policy will use duplex buffering. Defaults to False.
            info_mem_node (bool, optional): If True, the policy will return memory node in info. Defaults to False.
            independent_switches (bool, optional): If True, the policy will switch independently for each thread in batch. Defaults to False.
        """
        self._info_spec = info_spec
        self.duplex_buffering = False

        self.info_mem_node = info_mem_node
        if switch_probability is not None and switch_probability > 0.0:
            self.switch_probability = switch_probability
            self.independent_switches = independent_switches
            if independent_switches:
                self.switched = tf.zeros(
                    (tf_environment.batch_size, 1), dtype=tf.bool)
                policy_state_spec = {"fsc_state": TensorSpec(shape=(), dtype=tf.int32),
                                     "switched": TensorSpec(shape=(), dtype=tf.bool),
                                     "ppo_state": parallel_policy.policy_state_spec}
            else:
                self.switched = False
                policy_state_spec = {
                    "fsc_state": TensorSpec(shape=(), dtype=tf.int32),
                    "switched": TensorSpec(shape=(), dtype=tf.bool),
                    "ppo_state": parallel_policy.policy_state_spec
                }
        else:
            policy_state_spec = TensorSpec(shape=(), dtype=tf.int32)

        super(FSC_Policy, self).__init__(tf_environment._time_step_spec, tf_environment._action_spec,
                                         policy_state_spec=policy_state_spec,
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
        self.tf_action_labels = tf.constant(
            tf_action_keywords, dtype=tf.string)
        self._parallel_policy = parallel_policy
        if parallel_policy is not None:
            self._parallel_policy_function = common.function(
                parallel_policy.action)
            self._hidden_ppo_state = self._parallel_policy.get_initial_state(
                tf_environment.batch_size)
        self._soft_decision = soft_decision
        self._fsc_update_coef = soft_decision_multiplier

        if switch_probability is not None:
            self.switching = True
        else:
            self.switching = False

    def init_duplex_buffering(self, original_info_spec):
        self.duplex_buffering = True
        self._info_spec = {
            'fsc': TensorSpec(shape=(1,), dtype=tf.bool, name='fsc'),
            'rl': original_info_spec,
            'mem_node': TensorSpec(shape=(1,), dtype=tf.int32, name='mem_node')
        }

    def _set_hidden_ppo_state(self, batch_size=1):
        if self._parallel_policy is not None:
            self._hidden_ppo_state = self._parallel_policy.get_initial_state(
                batch_size)

    def _get_initial_state(self, batch_size):
        if not self.switching:
            self._set_hidden_ppo_state(batch_size=batch_size)
            self.switched = False
            return tf.zeros((batch_size, 1), dtype=tf.int32)
        else:
            return {"fsc_state": tf.zeros((batch_size, 1), dtype=tf.int32),
                    "switched": tf.zeros((batch_size, 1), dtype=tf.bool),
                    "ppo_state": self._parallel_policy.get_initial_state(batch_size)}

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

        tf_action_numbers = tf.map_fn(
            map_action_number, action_numbers, dtype=tf.int32)
        return tf_action_numbers

    def _make_soft_decision(self, fsc_action_number, time_step, seed):
        distribution = self._parallel_policy.distribution(
            time_step, self._hidden_ppo_state)
        self._hidden_ppo_state = distribution.state
        policy_info = distribution.info
        logits = distribution.action.logits
        one_hot_encoding = tf.one_hot(fsc_action_number, len(
            self.tf_action_labels)) * self._fsc_update_coef
        updated_logits = logits + one_hot_encoding
        action_number = tfp.distributions.Categorical(
            logits=updated_logits).sample()[0]
        return action_number, policy_info

    def _generate_paynt_decision(self, time_step, policy_state, seed):
        observation = time_step.observation["integer"]
        int_policy_state = tf.squeeze(policy_state)
        observation = tf.squeeze(observation)
        indices = tf.stack([int_policy_state, observation], axis=1)
        action_number = tf.gather_nd(self._fsc.action_function, indices)
        action_number = self.convert_to_tf_action_number(action_number)
        new_policy_state = tf.gather_nd(self._fsc.update_function, indices)
        new_policy_state = tf.convert_to_tensor(tf.reshape(
            new_policy_state, shape=(-1, 1)), dtype=tf.int32)
        return action_number, new_policy_state

    def _create_one_hot_fake_info(self, action_number):
        one_hot_encoding = tf.one_hot(
            action_number, len(self.tf_action_labels)) / 2.0
        one_hot_encoding_with_alternative = tf.where(
            one_hot_encoding == 0.0, -1.0, one_hot_encoding)
        one_hot_encoding = tf.cast(one_hot_encoding_with_alternative,
                                   tf.float32, name="CategoricalProjectionNetwork_logits")
        return {"dist_params": {"logits": one_hot_encoding}}

    def _get_switched_ppo_action(self, time_step, seed):
        parallel_policy_step = self._parallel_policy_function(
            time_step, self._hidden_ppo_state, seed)
        self._hidden_ppo_state = parallel_policy_step.state
        return parallel_policy_step.action, parallel_policy_step.info

    def get_reinited_hidden_ppo_state(self, hidden_ppo_state, first_step_mask):
        init_state = self._parallel_policy.get_initial_state(
            tf.shape(first_step_mask)[0])
        for key in self._hidden_ppo_state.keys():
            for i in range(len(self._hidden_ppo_state[key])):
                self._hidden_ppo_state[key][i] = tf.where(
                    first_step_mask,
                    init_state[key][i],
                    self._hidden_ppo_state[key][i]
                )
        return self._hidden_ppo_state

    def get_reinited_policy_state(self, policy_state, first_step_mask):
        policy_state = tf.where(
            first_step_mask,
            self._get_initial_state(tf.shape(first_step_mask)[0]),
            policy_state
        )
        return policy_state

    def _independent_switching_action(self, time_step, policy_state, seed) -> PolicyStep:
        # Generate switches for each thread in batch given the probability of switching
        batch_size = tf.shape(time_step.observation["integer"])[0]
        switch_mask = tf.random.uniform(
            (batch_size,), minval=0, maxval=1, dtype=tf.float32) < self.switch_probability
        
        # Set policy states that were not switched yet
        self.switched = tf.where(switch_mask, tf.ones(batch_size, ), self.switched)

        # Get initial states for each switched thread
        initial_state = self._parallel_policy.get_initial_state(batch_size)
        policy_state["ppo_state"] = tf.where(
            switch_mask,
            initial_state,
            policy_state["ppo_state"]
        )
        policy_state["switched"] = self.switched
        action_number_fsc, new_policy_state_fsc = self._generate_paynt_decision(
            time_step, policy_state["fsc_state"], seed)
        policy_state["fsc_state"] = new_policy_state_fsc
        policy_step_ppo = self._parallel_policy_function(
            time_step, policy_state["ppo_state"], seed)
        policy_state["ppo_state"] = policy_step_ppo.state
        policy_info = policy_step_ppo.info
        action_number_ppo = policy_step_ppo.action
        action_number = tf.where(
            self.switched,
            action_number_ppo,
            action_number_fsc
        )
        return PolicyStep(action_number, policy_state, policy_info)

    def _dependent_switching_action(self, time_step, policy_state, seed) -> PolicyStep:
        if not self.switched:
            switch_mask = tf.random.uniform((), minval=0, maxval=1, dtype=tf.float32) < self.switch_probability
            policy_state["ppo_state"] = self._parallel_policy.get_initial_state(tf.shape(time_step.observation["integer"])[0])
            self.switched = switch_mask
            policy_state["switched"] = tf.ones(tf.shape(time_step.observation["integer"])[0], dtype=tf.bool)
        if self.switched:
            policy_step = self._parallel_policy_function(
                time_step, policy_state["ppo_state"], seed)
            policy_state["ppo_state"] = policy_step.state
            return PolicyStep(policy_step.action, policy_state, policy_step.info)
        else:
            action_number, new_policy_state = self._generate_paynt_decision(
                time_step, policy_state["fsc_state"], seed)
            policy_state["fsc_state"] = new_policy_state
            return PolicyStep(action_number, policy_state, self._create_one_hot_fake_info(action_number))
        
    def reset_switching(self):
        self.switched = False

    @tf.function
    def _action(self, time_step, policy_state, seed) -> PolicyStep:
        # Change the policy_state for the first time steps
        if self.switching and self.independent_switches:
            return self._independent_switching_action(time_step, policy_state, seed)
        elif self.switching:
            return self._dependent_switching_action(time_step, policy_state, seed)
        equal_mask = tf.math.equal(time_step.step_type, StepType.FIRST)
        equal_mask = tf.reshape(equal_mask, shape=(-1, 1))
        policy_state = self.get_reinited_policy_state(policy_state, equal_mask)
        self._hidden_ppo_state = self.get_reinited_hidden_ppo_state(
            self._hidden_ppo_state, equal_mask)
        batch_size = tf.shape(time_step.observation["integer"])[0]
        policy_info = ()
        action_number, new_policy_state = self._generate_paynt_decision(
            time_step, policy_state, seed)
        if self._info_spec is None or self._info_spec == ():
            policy_info = ()

        elif self._parallel_policy is not None and policy_info == ():  # Generate logits from PPO policy
            if self._soft_decision:  # Use PPO policy to make a decision combined with FSC
                action_number, policy_info = self._make_soft_decision(
                    action_number, time_step, seed)
            else:  # Hard FSC decision
                parallel_policy_step = self._parallel_policy_function(
                    time_step, self._hidden_ppo_state, seed)
                self._hidden_ppo_state = parallel_policy_step.state
                policy_info = parallel_policy_step.info
        if self.duplex_buffering:
            policy_info = {
                "fsc": tf.constant([not self.switched] * batch_size, dtype=tf.bool),
                "rl": policy_info,
                "mem_node": new_policy_state
            }
        if policy_info == () and self.info_mem_node:
            policy_info = {"mem_node": new_policy_state}
        # If parallel policy does not return logits, use one-hot encoding of action number
        elif policy_info == () and self._info_spec != ():
            policy_info = self._create_one_hot_fake_info(action_number)
        policy_step = PolicyStep(action=action_number,
                                 state=new_policy_state, info=policy_info)
        return policy_step
