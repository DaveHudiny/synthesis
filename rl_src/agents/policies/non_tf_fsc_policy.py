from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep
from agents.policies.fsc_copy import FSC

from tf_agents.policies import TFPolicy
from tf_agents.specs.tensor_spec import TensorSpec

import tensorflow as tf

import numpy as np

import tqdm


def convert_to_tf_action_number(action_numbers, original_action_labels, tf_action_labels):
    # @tf.function
    def map_action_number(action_number):
        keyword = original_action_labels[action_number]
        if "__no_label__" == keyword:
           return tf.constant(-1, dtype=tf.int32)
        
        tf_action_number = tf.argmax(
            tf.cast(tf.equal(tf_action_labels, keyword), tf.int32), output_type=tf.int32)
        return tf_action_number

    tf_action_numbers = tf.map_fn(
        map_action_number, action_numbers, dtype=tf.int32)
    return tf_action_numbers


def fsc_action_constraint_splitter(observation):
    return observation["observation"], observation["mask"], observation["integer"]


class NonTFFSCPolicy(TFPolicy):
    def __init__(self, fsc: FSC, tf_action_keywords, time_step_spec, action_spec, policy_state_spec=(), info_spec=(), name=None,
                 observation_and_action_constraint_splitter=None, fsc_action_keywords : list[str] = None):

        if policy_state_spec != ():
            raise NotImplementedError(
                "PAYNT currently only supports FSC policies with a single integer state")
        policy_state_spec = TensorSpec(shape=(), dtype=tf.int32)
        self.is_stochastic = not fsc.is_deterministic
        
        super(NonTFFSCPolicy, self).__init__(time_step_spec, action_spec, policy_state_spec=policy_state_spec, info_spec=info_spec, name=name,
                                              observation_and_action_constraint_splitter=observation_and_action_constraint_splitter)
        self.tf_action_labels = tf.constant(tf_action_keywords, dtype=tf.string)
        self._fsc = fsc
        self._fsc.action_labels = tf.constant(fsc.action_labels, dtype=tf.string)
        # Initialize the step counter tqdm bar
        self.step_counter = tqdm.tqdm(total=0, position=0, leave=True)
        self.steps = 0
        
        self.num_actions = len(self._fsc.action_labels)

    def _get_initial_state(self, batch_size):
        return tf.zeros((batch_size, 1), dtype=tf.int32)

    def sample_from_dict(self, dictus, nr_options):
        """
        Sample a number from a dictionary of probabilities with shape {option: probability}
        
        :param dictus: dictionary with options as keys and probabilities as values
        :param nr_options: number of options to sample from
        :return: sampled option number
        """ 
        # Check whether dictus is a dictionary and not a single scalar
        if type(dictus) is not dict:
            print("Warning: dictus is not a dictionary, returning 0")
            return dictus if isinstance(dictus, int) else 0
        options = np.array(list(dictus.keys()), dtype=np.int32)
        probabilities = np.array(list(dictus.values()), dtype=np.float32)
        probabilities_complete = np.zeros((nr_options,), dtype=np.float32)
        probabilities_complete[options] = probabilities
        normalized_probabilities = probabilities_complete / np.sum(probabilities_complete)
        sampled_option = np.random.choice(
            nr_options, p=normalized_probabilities)
        return sampled_option


    def get_action_and_update(self, policy_state, observation_integer, is_stochastic):
        fsc_action_numbers = np.zeros((len(observation_integer),), dtype=np.int32)
        fsc_update_numbers = np.zeros((len(observation_integer),), dtype=np.int32)
        for i, sub_batch in enumerate(zip(observation_integer, policy_state)):
            if self._fsc.is_deterministic:
                fsc_action_numbers[i] = self._fsc.action_function[sub_batch[1]][sub_batch[0]]
                fsc_update_numbers[i] = self._fsc.update_function[sub_batch[1]][sub_batch[0]]
            else:
                probs_action = self._fsc.action_function[sub_batch[1]][sub_batch[0]]
                probs_update = self._fsc.update_function[sub_batch[1]][sub_batch[0]]
                fsc_action_numbers[i] = self.sample_from_dict(
                    probs_action, self.num_actions)
                fsc_update_numbers[i] = self.sample_from_dict(
                    probs_update, self._fsc.num_nodes)

        tf_action_numbers = convert_to_tf_action_number(
                fsc_action_numbers, self._fsc.action_labels, self.tf_action_labels)
        fsc_update_numbers = tf.convert_to_tensor(
            fsc_update_numbers, dtype=tf.int32)
        fsc_update_numbers = tf.reshape(fsc_update_numbers, (-1, 1))
        return tf_action_numbers, fsc_update_numbers

    def _action(self, time_step: TimeStep, policy_state, seed):
        self.steps = self.steps + 1
        _, _, integer = fsc_action_constraint_splitter(time_step.observation)
        integer = tf.squeeze(integer).numpy()
        policy_state = tf.squeeze(policy_state).numpy()
        action_number, new_policy_state = self.get_action_and_update(
            policy_state, integer, self.is_stochastic)
        return PolicyStep(action_number, new_policy_state, ())
