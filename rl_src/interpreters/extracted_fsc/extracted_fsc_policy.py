from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.policy_step import PolicyStep
import tensorflow as tf


class ExtractedFSCPolicy(TFPolicy):
    def __init__(self, time_step_spec, action_spec, policy_state_spec, observation_to_action_table=None, observation_to_update_table=None, labels=None, memory_size=1, observation_size=1):
        super(ExtractedFSCPolicy, self).__init__(
            time_step_spec, action_spec, policy_state_spec)
        if observation_to_action_table is not None:
            self.observation_to_action_table = tf.constant(
                observation_to_action_table, dtype=tf.int32)
        else:
            self.tf_observation_to_action_table = tf.zeros(
                shape=(memory_size, observation_size), dtype=tf.int32)
        if observation_to_update_table is not None:
            self.observation_to_update_table = tf.constant(
                observation_to_update_table, dtype=tf.int32)
        else:
            self.tf_observation_to_update_table = tf.zeros(
                shape=(memory_size, observation_size), dtype=tf.int32)
        self.model_memory_size = 0
        self.action_labels = labels

    @tf.function
    def _action(self, time_step: TimeStep, policy_state: PolicyStep, seed):
        observation = time_step.observation["integer"]
        # if self.model_memory_size == 0:
        #     memory = policy_state
        # else:
        #     memory = tf.cast(
        #         time_step.observation["observation"][:, -1], dtype=tf.int32)
        #     memory = tf.reshape(memory, (-1, 1))
        memory = policy_state
        indices = tf.concat([memory, observation], axis=1)
        action = tf.gather_nd(self.tf_observation_to_action_table, indices)
        update = tf.gather_nd(self.tf_observation_to_update_table, indices)
        update = tf.reshape(update, (-1, 1))
        # action = self.observation_to_action_table[memory, observation]
        # update = self.observation_to_update_table[memory, observation]
        if self.model_memory_size == 0:
            action_dict = action
            policy_state = update
        else:
            action_dict = {
                "simulator_action": action,
                "memory_update": update
            }

        return PolicyStep(action_dict, policy_state)

    def _get_initial_state(self, batch_size):
        return tf.zeros((batch_size, 1), dtype=tf.int32)
