from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep
from agents.policies.fsc_copy import FSC

from tf_agents.policies import TFPolicy
from tf_agents.specs.tensor_spec import TensorSpec

import tensorflow as tf

import numpy as np

import tqdm


def convert_to_tf_action_number(action_numbers, original_action_labels, tf_action_labels):
    @tf.function
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


class SimpleFSCPolicy(TFPolicy):
    def __init__(self, fsc: FSC, tf_action_keywords, time_step_spec, action_spec, policy_state_spec=(), info_spec=(), name=None,
                 observation_and_action_constraint_splitter=None, fsc_action_keywords : list[str] = None):

        if policy_state_spec != ():
            raise NotImplementedError(
                "PAYNT currently only supports FSC policies with a single integer state")
        policy_state_spec = TensorSpec(shape=(), dtype=tf.int32)
        self.is_stochastic = not fsc.is_deterministic
        
        super(SimpleFSCPolicy, self).__init__(time_step_spec, action_spec, policy_state_spec=policy_state_spec, info_spec=info_spec, name=name,
                                              observation_and_action_constraint_splitter=observation_and_action_constraint_splitter)
        self.init_fsc_to_tf(fsc, tf_action_keywords, self.is_stochastic, fsc_action_keywords)
        # Initialize the step counter tqdm bar
        self.step_counter = tqdm.tqdm(total=0, position=0, leave=True)
        self.steps = 0
        
    def convert_to_sparse_tf(self, table_function) -> tf.SparseTensor:
        max_action = 0
        actions = []
        indices = []
        
        for memory_int, memory in enumerate(table_function):
            for observation_int, observation in enumerate(memory):
                if observation is not None:
                    for action in observation:
                        prob = observation[action]
                        actions.append(prob)
                        indices.append([memory_int, observation_int, action])
                        if action > max_action:
                            max_action = action
                else:
                    indices.append([memory_int, observation_int, 0])
                    actions.append(1.0)
        
        sparse_probs = tf.SparseTensor(indices, actions, [len(table_function), len(table_function[0]), max_action + 1])
        # dense_probs = tf.sparse.to_dense(sparse_probs)
        return sparse_probs
    
    def create_inference_tensors(self, table_function, is_update_function=False):
        max_action = self._fsc.action_labels.shape[0] if not is_update_function else len(table_function[0])
        is_det_table = np.ones((len(table_function), len(table_function[0])), dtype=bool)
        det_choice_table = np.zeros((len(table_function), len(table_function[0])), dtype=np.int32)
        non_det_choice_table = []
        non_det_action_probs_table = []
        observations_number = 0
        observation_miss_number = 0
        for memory_int, memory in enumerate(table_function):
            for observation_int, observation in enumerate(memory):
                observations_number += 1
                if observation is not None:
                    
                    actions = np.array(list(observation.keys()))
                    probs = np.array(list(observation.values()))
                    if len(actions) == 1:
                        det_choice_table[memory_int, observation_int] = actions[0]
                    else:
                        is_det_table[memory_int, observation_int] = False
                        probs_vec = np.zeros((max_action,))
                        probs_vec[actions] = probs
                        non_det_choice_table.append([memory_int, observation_int])
                        non_det_action_probs_table.append(probs_vec)
                else:
                    observation_miss_number += 1
                    is_det_table[memory_int, observation_int] = True
                    det_choice_table[memory_int, observation_int] = 0
        # Initialize lookup tables for non-deterministic choices
        non_det_choice_table = np.array(non_det_choice_table)
        non_det_action_probs_table = np.array(non_det_action_probs_table)
        if non_det_choice_table.shape[0] == 0:
            lookup_table = None
        else:
            keys = tf.strings.as_string(non_det_choice_table[:, 0]) + "_" + tf.strings.as_string(non_det_choice_table[:, 1])
            lookup_table = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    keys,
                    tf.range(len(non_det_choice_table), dtype=tf.int32),
                    key_dtype=tf.string,
                    value_dtype=tf.int32
                ),
                default_value=-1
            )
        

        return is_det_table, det_choice_table, lookup_table, non_det_action_probs_table

    def check_same_types(self, a : list[list[dict[int, float]]], b : list[list[dict[int, float]]]):
        if len(a) != len(b):

            return False
        if type(a) != type(b):
            print(f"Type mismatch: {type(a)} != {type(b)}")
            return False
        for i in range(len(a)):
            if type(a[i]) != type(b[i]):
                print(f"Type mismatch at row {i}: {type(a[i])} != {type(b[i])}")
                return False
            if len(a[i]) != len(b[i]):
                print(f"Length mismatch at row {i}: {len(a[i])} != {len(b[i])}")
                return False
            for j in range(len(a[i])):
                if a[i][j] is None and b[i][j] is None:
                    continue
                if a[i][j] is None or b[i][j] is None:
                    print(f"Type mismatch at ({i}, {j}): {a[i][j]} != {b[i][j]}")
                    return False
                for k, l in zip(a[i][j].keys(), b[i][j].keys()):
                    if type(a[i][j][k]) != type(b[i][j][l]):
                        print(f"Type mismatch at ({i}, {j}, {k}): {type(a[i][j][k])} != {type(b[i][j][l])}")
                        return False
        return True
    
    def check_create_inference_tensors_for_a_b(self, a : list[list[dict[int, float]]], b : list[list[dict[int, float]]]):
        is_det_a, det_choice_a, look_up_a, non_det_prob_a = self.create_inference_tensors(a)
        is_det_b, det_choice_b, look_up_b, non_det_prob_b = self.create_inference_tensors(b)
        if not np.array_equal(is_det_a, is_det_b):
            print("is_det_table mismatch")
            return False
        if not np.array_equal(det_choice_a, det_choice_b):
            print("det_choice_table mismatch")
            return False
        if look_up_a is None and look_up_b is None:
            return True
        if look_up_a is None or look_up_b is None:
            print("look_up_table mismatch")
            return False
        if not tf.reduce_all(tf.equal(look_up_a.lookup(look_up_a.keys()), look_up_b.lookup(look_up_b.keys()))):
            print("look_up_table keys mismatch")
            return False
        if not tf.reduce_all(tf.equal(non_det_prob_a, non_det_prob_b)):
            print("non_det_action_probs_table mismatch")
            return False
        return True

    def init_fsc_to_tf(self, fsc: FSC, tf_action_keywords : list[str], is_stochastic, original_action_keywords_order: list[str] = None):
        self._fsc = fsc
        self._fsc.action_labels = tf.constant(
                self._fsc.action_labels, dtype=tf.string)
        self.tf_action_labels = tf.constant(
                tf_action_keywords, dtype=tf.string)
            

        # Remap the action in action_function given the fsc_action_keywords ()

        if not is_stochastic:
            array_action_function = np.array(self._fsc.action_function, dtype=np.int32)
            self._fsc.action_function = tf.constant(
                array_action_function, dtype=tf.int32)
            self._fsc.update_function = tf.constant(
                self._fsc.update_function, dtype=tf.int32)
            
            
        else: # action_funciton contains from dicts of probabilities for each action. The shape (without dict) is [memory_size, observation_size]. Dicts are sparse representation of the action function distribution.
            # self.tf_sparse_action_function = self.convert_to_sparse_tf(self._fsc.action_function)
            # self.tf_sparse_update_function = self.convert_to_sparse_tf(self._fsc.update_function)
            is_det_t, det_ch_t, look_up_t, non_det_prob_t = self.create_inference_tensors(self._fsc.action_function)
            is_det_u, det_ch_u, look_up_u, non_det_prob_u = self.create_inference_tensors(self._fsc.update_function, is_update_function=True)
            self.is_det_table_action = tf.constant(is_det_t, dtype=tf.bool)
            self.det_choice_table_action = tf.constant(det_ch_t, dtype=tf.int32)
            self.non_det_choice_table_action = look_up_t
            self.non_det_action_probs_table_action = tf.constant(non_det_prob_t, dtype=tf.float32)
            self.is_det_table_update = tf.constant(is_det_u, dtype=tf.bool)
            self.det_choice_table_update = tf.constant(det_ch_u, dtype=tf.int32)
            self.non_det_choice_table_update = look_up_u
            self.non_det_action_probs_table_update = tf.constant(non_det_prob_u, dtype=tf.float32)
            # raise "Debugging"
                    


    def _get_initial_state(self, batch_size):
        return tf.zeros((batch_size, 1), dtype=tf.int32)

    def _distribution(self, time_step: TimeStep, policy_state, seed) -> PolicyStep:
        raise NotImplementedError(
            "PAYNT currently implements only deterministic FSC policies")

    @tf.function
    def _action_number(self, policy_state, observation_integer, is_stochastic):
        indices = tf.stack([policy_state, observation_integer], axis=1)
        indices = tf.cast(indices, dtype=tf.int64)
        if is_stochastic:
            is_det_choice = tf.gather_nd(self.is_det_table_action, indices)
            det_choice = tf.gather_nd(self.det_choice_table_action, indices)
            
            if self.non_det_choice_table_action is None:

                fsc_action_numbers = tf.where(is_det_choice, det_choice, 0)
                fsc_action_numbers = tf.reshape(fsc_action_numbers, shape=(-1,))
            else:
                # Find non-det indices from the lookup table
                str_indices = tf.strings.as_string(indices[:, 0]) + "_" + tf.strings.as_string(indices[:, 1])
                non_det_indices = self.non_det_choice_table_action.lookup(str_indices)
                # Remove -1 indices
                non_det_indices = tf.where(non_det_indices != -1)
                non_det_indices = tf.cast(non_det_indices, dtype=tf.int32)
                # Convert non-det-indices to self.non_det_action_probs_table_action indices by self.non_det_choice_table_action
                default_probs = tf.zeros([tf.shape(is_det_choice)[0], self._fsc.action_labels.shape[0]], dtype=tf.float32)
                non_det_probs = tf.gather_nd(self.non_det_action_probs_table_action, non_det_indices)
                # Replace default probs with non_det_probs given non_det_indices
                non_det_probs = tf.tensor_scatter_nd_update(default_probs, non_det_indices, non_det_probs)
                det_choice = tf.reshape(det_choice, shape=(-1, 1))
                is_det_choice = tf.reshape(is_det_choice, shape=(-1, 1))
                fsc_action_numbers = tf.where(is_det_choice, det_choice, tf.random.categorical(
                    tf.math.log(non_det_probs), 1, dtype=tf.int32))
                fsc_action_numbers = tf.reshape(fsc_action_numbers, shape=(-1,))

        else:
            fsc_action_numbers = tf.gather_nd(self._fsc.action_function, indices)
        tf_action_numbers = convert_to_tf_action_number(
                fsc_action_numbers, self._fsc.action_labels, self.tf_action_labels)

        return tf_action_numbers

    # @tf.function
    def _new_fsc_state(self, policy_state, observation_integer, is_stochastic):
        indices = tf.stack([policy_state, observation_integer], axis=1)
        indices = tf.cast(indices, dtype=tf.int64)
        if is_stochastic:
            is_det_choice = tf.gather_nd(self.is_det_table_update, indices)
            det_choice = tf.gather_nd(self.det_choice_table_update, indices)
            new_policy_state = tf.where(is_det_choice, det_choice, 0)
            new_policy_state = tf.reshape(new_policy_state, shape=(-1,))
        else:
            new_policy_state = tf.gather_nd(self._fsc.update_function, indices)
        new_policy_state = tf.convert_to_tensor(tf.reshape(
            new_policy_state, shape=(-1, 1)), dtype=tf.int32)
        
        return new_policy_state

    def _action(self, time_step: TimeStep, policy_state, seed):
        self.steps = self.steps + 1
        _, _, integer = fsc_action_constraint_splitter(time_step.observation)
        integer = tf.squeeze(integer)
        policy_state = tf.squeeze(policy_state)
        action_number = self._action_number(policy_state, integer, is_stochastic=self.is_stochastic)
        new_policy_state = self._new_fsc_state(policy_state, integer, is_stochastic=self.is_stochastic)
        return PolicyStep(action_number, new_policy_state, ())
