from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.policy_step import PolicyStep
import tensorflow as tf
from tf_agents.specs.tensor_spec import TensorSpec
import numpy as np

from interpreters.extracted_fsc.extracted_fsc_policy import ExtractedFSCPolicy

class TableBasedPolicy(TFPolicy):
    def __init__(self, original_policy : TFPolicy, action_function : np.ndarray, 
                 update_function : np.ndarray, # Action and update function has shape (nr_model_states, nr_observations) 
                 initial_memory = 0,
                 action_keywords = None,
                 descending_actions = None
                 ):
        """
        TableBasedPolicy is a policy that uses a table to map observations to actions and updates.
        
        Args:
            original_policy (TFPolicy): The original policy to be used. (TODO: Remove this, since you can derive the policy from the model.)
            action_function (np.ndarray): The action function table. It has shape (nr_model_states, nr_observations).
            update_function (np.ndarray): The update function table. It has shape (nr_model_states, nr_observations)."""
        policy_state_spec = TensorSpec(shape=(), dtype=tf.int32)
        super(TableBasedPolicy, self).__init__(original_policy.time_step_spec, original_policy.action_spec, policy_state_spec=policy_state_spec)
        self.tf_observation_to_action_table = tf.constant(action_function, dtype=tf.int32)
        self.tf_observation_to_update_table = tf.constant(update_function, dtype=tf.int32)
        if descending_actions is not None: # This array contains actions for a given memory and observation in descending order
                                           # The first action is the most likely one, but it could be illegal in some cases
                                           # This array is used to get the most likely legal action, when combined mask
            self.descending_actions = tf.constant(descending_actions, dtype=tf.int32)
        else:
            self.descending_actions = None
        
        self.action_keywords = action_keywords
        self.initial_memory = initial_memory

    def _get_initial_state(self, batch_size):
        return tf.constant(self.initial_memory, shape=(batch_size, 1), dtype=tf.int32)

    @tf.function
    def _action(self, time_step, policy_state, seed):
        observation = time_step.observation["integer"]
        mask = time_step.observation["mask"]
        memory = policy_state
        indices = tf.concat([memory, observation], axis=1)
        if self.descending_actions is not None:
            # Gather the ordered actions for the current observation, remove the illegal actions and select the first legal action
            descending_actions = tf.gather_nd(self.descending_actions, indices)
            descending_actions = tf.boolean_mask(descending_actions, mask)
            action = descending_actions[0]
        else:
            action = tf.gather_nd(self.tf_observation_to_action_table, indices)
        update = tf.gather_nd(self.tf_observation_to_update_table, indices)
        update = tf.reshape(update, (-1, 1))

        policy_step = PolicyStep(action, update)
        return policy_step