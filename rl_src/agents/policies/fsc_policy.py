# Includes implemenation of FSCPolicy class, which is a TFPolicy implementation of FSC used for Hard and Soft FSC oracle.
# Author: David Hudák
# Login: xhudak03
# File: fsc_policy.py

from tools.encoding_methods import *
from tf_agents.environments import tf_py_environment
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.specs.tensor_spec import TensorSpec
from tf_agents.utils import common

import tensorflow_probability as tfp

import tensorflow as tf


import json

import logging
logger = logging.getLogger(__name__)


class FSC:
    '''
    Class for encoding an FSC having
    - a fixed number of nodes
    - action selection is either:
        + deterministic: gamma: NxZ -> Act, or
        + randomized: gamma: NxZ -> Distr(Act), where gamma(n,z) is a dictionary of pairs (action,probability)
    - deterministic posterior-unaware memory update delta: NxZ -> N
    '''

    def __init__(self, num_nodes, num_observations, is_deterministic=False):
        self.num_nodes = num_nodes
        self.num_observations = num_observations
        self.is_deterministic = is_deterministic

        self.action_function = [
            [None]*num_observations for _ in range(num_nodes)]
        self.update_function = [
            [None]*num_observations for _ in range(num_nodes)]

        self.observation_labels = None
        self.action_labels = None

    def __str__(self):
        return json.dumps(self.to_json(), indent=4)

    def action_function_signature(self):
        if self.is_deterministic:
            return " NxZ -> Act"
        else:
            return " NxZ -> Distr(Act)"

    def to_json(self):
        json = {}
        json["num_nodes"] = self.num_nodes
        json["num_observations"] = self.num_observations
        json["__comment_action_function"] = "action function has signature {}".format(
            self.action_function_signature())
        json["__comment_update_function"] = "update function has signature NxZ -> N"

        json["action_function"] = self.action_function
        json["update_function"] = self.update_function

        if self.action_labels is not None:
            json["action_labels"] = self.action_labels
        if self.observation_labels is not None:
            json["observation_labels"] = self.observation_labels

        return json

    @classmethod
    def from_json(cls, json):
        num_nodes = json["num_nodes"]
        num_observations = json["num_observations"]
        fsc = FSC(num_nodes, num_observations)

        fsc.action_function = json["action_function"]
        fsc.update_function = json["update_function"]

        fsc.action_labels = json["action_labels"] if "action_labels" in json else None
        fsc.observation_labels = json["observation_labels"] if "observation_labels" in json else None
        return fsc

    def check_action_function(self, observation_to_actions):
        assert len(
            self.action_function) == self.num_nodes, "FSC action function is not defined for all memory nodes"
        for node in range(self.num_nodes):
            assert len(self.action_function[node]) == self.num_observations, \
                "in memory node {}, FSC action function is not defined for all observations".format(
                    node)
            for obs in range(self.num_observations):
                if self.is_deterministic:
                    action = self.action_function[node][obs]
                    assert action in observation_to_actions[obs], "in observation {} FSC chooses invalid action {}".format(
                        obs, action)
                else:
                    for action, _ in self.action_function[node][obs].items():
                        assert action in observation_to_actions[obs], "in observation {} FSC chooses invalid action {}".format(
                            obs, action)

    def check_update_function(self):
        assert len(
            self.update_function) == self.num_nodes, "FSC update function is not defined for all memory nodes"
        for node in range(self.num_nodes):
            assert len(self.update_function[node]) == self.num_observations, \
                "in memory node {}, FSC update function is not defined for all observations".format(
                    node)
            for obs in range(self.num_observations):
                update = self.update_function[node][obs]
                assert 0 <= update and update < self.num_nodes, "invalid FSC memory update {}".format(
                    update)

    def check(self, observation_to_actions):
        ''' Check whether fields of FSC have been initialized appropriately. '''
        assert self.num_nodes > 0, "FSC must have at least 1 node"
        self.check_action_function(observation_to_actions)
        self.check_update_function()

    def fill_trivial_actions(self, observation_to_actions):
        ''' For each observation with 1 available action, set gamma(n,z) to that action. '''
        for obs, actions in enumerate(observation_to_actions):
            if len(actions) > 1:
                continue
            action = actions[0]
            if not self.is_deterministic:
                action = {action: 1}
            for node in range(self.num_nodes):
                self.action_function[node][obs] = action

    def fill_trivial_updates(self, observation_to_actions):
        ''' For each observation with 1 available action, set delta(n,z) to n. '''
        for obs, actions in enumerate(observation_to_actions):
            if len(actions) > 1:
                continue
            for node in range(self.num_nodes):
                self.update_function[node][obs] = node

    def fill_zero_updates(self):
        for node in range(self.num_nodes):
            self.update_function[node] = [0] * self.num_observations


class FSC_Policy(TFPolicy):
    def __init__(self, tf_environment: tf_py_environment.TFPyEnvironment, fsc: FSC,
                 observation_and_action_constraint_splitter=None, tf_action_keywords=[],
                 info_spec=None, parallel_policy : TFPolicy = None, soft_decision = False,
                 soft_decision_multiplier : float = 2.0, need_logits : bool = True):
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
        super(FSC_Policy, self).__init__(tf_environment._time_step_spec, tf_environment._action_spec,
                                         policy_state_spec=TensorSpec(
                                             shape=(), dtype=tf.int32),
                                         observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
                                         info_spec=info_spec,
                                        )
        self._time_step_spec = tf_environment._time_step_spec
        self._action_spec = tf_environment._action_spec
        self._observation_and_action_constraint_splitter = observation_and_action_constraint_splitter
        self._info_spec = info_spec
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
            self._hidden_ppo_state = self._parallel_policy.get_initial_state(1)
        self._soft_decision = soft_decision
        self._fsc_update_coef = soft_decision_multiplier

    tf.function
    def _set_hidden_ppo_state(self):
        if self._parallel_policy is not None:
            self._hidden_ppo_state = self._parallel_policy.get_initial_state(1)

    def _get_initial_state(self, batch_size):
        self._set_hidden_ppo_state()
        return tf.constant([0], dtype=tf.int32)

    def _distribution(self, time_step, policy_state):
        raise NotImplementedError(
            "FSC policy does not support distribution-based action selection")

    @tf.function
    def convert_to_tf_action_number(self, action_number):
        keyword = self._fsc_action_labels[action_number]
        if keyword == "__no_label__":
            return tf.constant(-1, dtype=tf.int32)
        tf_action_number = tf.argmax(
            tf.cast(tf.equal(self.tf_action_labels, keyword), tf.int32), output_type=tf.int32)
        return tf_action_number
    
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
        int_policy_state = tf.squeeze(policy_state[0])
        observation = tf.squeeze(observation)
        action_number = self._fsc.action_function[int_policy_state][observation]
        action_number = self.convert_to_tf_action_number(action_number)
        new_policy_state = self._fsc.update_function[int_policy_state][observation]

        return action_number, new_policy_state

    def _create_one_hot_fake_info(self, action_number):
        one_hot_encoding = tf.one_hot(action_number, len(self.tf_action_labels)) / 2.0
        one_hot_encoding_with_alternative = tf.where(one_hot_encoding == 0.0, -1.0, one_hot_encoding)
        one_hot_encoding = tf.cast([one_hot_encoding_with_alternative], tf.float32, name="CategoricalProjectionNetwork_logits")
        return {"dist_params": {"logits": one_hot_encoding}}
    
    @tf.function
    def _action(self, time_step, policy_state, seed):
        action_number, new_policy_state = self._generate_paynt_decision(time_step, policy_state, seed)
        if self._info_spec is None or self._info_spec == ():
            policy_info = ()
        elif self._parallel_policy is not None: # Generate logits from PPO policy
            if self._soft_decision: # Use PPO policy to make a decision combined with FSC
                action_number, policy_info = self._make_soft_decision(action_number, time_step, seed)
            else: # Hard FSC decision
                parallel_policy_step = self._parallel_policy_function(time_step, self._hidden_ppo_state, seed)
                self._hidden_ppo_state = parallel_policy_step.state
                policy_info = parallel_policy_step.info
                print("Vygenerováno", policy_info)
        if policy_info == (): # If parallel policy does not return logits, use one-hot encoding of action number
            policy_info = self._create_one_hot_fake_info(action_number)
        
        policy_step = PolicyStep(action=tf.convert_to_tensor(
            [action_number], dtype=tf.int32), state=new_policy_state, info=policy_info)
        return policy_step