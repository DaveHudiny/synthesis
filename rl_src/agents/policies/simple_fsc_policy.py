from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep
from agents.policies.fsc_copy import FSC

from tf_agents.policies import TFPolicy
from tf_agents.specs.tensor_spec import TensorSpec

import tensorflow as tf


def convert_to_tf_action_number(action_numbers, original_action_labels, tf_action_labels):
    @tf.function
    def map_action_number(action_number):
        keyword = original_action_labels[action_number]
        if keyword == "__no_label__":
            return tf.constant(-1, dtype=tf.int32)
        tf_action_number = tf.argmax(
            tf.cast(tf.equal(tf_action_labels, keyword), tf.int32), output_type=tf.int32)
        return tf_action_number

    tf_action_numbers = tf.map_fn(
        map_action_number, action_numbers, dtype=tf.int32)
    return tf_action_numbers


def fsc_action_constraint_splitter(observation):
    return observation["observation"], observation["mask"], observation["integer"]


class SimpleFSCPolicy(TFPolicy):
    def __init__(self, fsc: FSC, tf_action_keywords, time_step_spec, action_spec, policy_state_spec=(), info_spec=(), name=None,
                 observation_and_action_constraint_splitter=None):

        if policy_state_spec != ():
            raise NotImplementedError(
                "PAYNT currently only supports FSC policies with a single integer state")
        policy_state_spec = TensorSpec(shape=(), dtype=tf.int32)

        self.init_fsc_to_tf(fsc, tf_action_keywords)
        super(SimpleFSCPolicy, self).__init__(time_step_spec, action_spec, policy_state_spec=policy_state_spec, info_spec=info_spec, name=name,
                                              observation_and_action_constraint_splitter=observation_and_action_constraint_splitter)

    def init_fsc_to_tf(self, fsc: FSC, tf_action_keywords):
        self._fsc = fsc
        self._fsc.action_function = tf.constant(
            self._fsc.action_function, dtype=tf.int32)
        self._fsc.update_function = tf.constant(
            self._fsc.update_function, dtype=tf.int32)
        self._fsc.action_labels = tf.constant(
            self._fsc.action_labels, dtype=tf.string)
        self.tf_action_labels = tf.constant(
            tf_action_keywords, dtype=tf.string)

    def _get_initial_state(self, batch_size):
        return tf.zeros((batch_size, 1), dtype=tf.int32)

    def _distribution(self, time_step: TimeStep, policy_state, seed) -> PolicyStep:
        raise NotImplementedError(
            "PAYNT currently implement only deterministic FSC policies")

    def _action_number(self, policy_state, observation_integer):
        indices = tf.stack([policy_state, observation_integer], axis=1)
        fsc_action_numbers = tf.gather_nd(self._fsc.action_function, indices)
        tf_action_numbers = convert_to_tf_action_number(
            fsc_action_numbers, self._fsc.action_labels, self.tf_action_labels)
        return tf_action_numbers

    def _new_fsc_state(self, policy_state, observation_integer):
        indices = tf.stack([policy_state, observation_integer], axis=1)
        new_policy_state = tf.gather_nd(self._fsc.update_function, indices)
        new_policy_state = tf.convert_to_tensor(tf.reshape(
            new_policy_state, shape=(-1, 1)), dtype=tf.int32)
        return new_policy_state

    def _action(self, time_step: TimeStep, policy_state, seed):
        _, _, integer = fsc_action_constraint_splitter(time_step.observation)
        integer = tf.squeeze(integer)
        policy_state = tf.squeeze(policy_state)
        action_number = self._action_number(policy_state, integer)
        new_policy_state = self._new_fsc_state(policy_state, integer)
        return PolicyStep(action_number, new_policy_state, ())
