from rl_src.agents.encoding_methods import *
from tf_agents.environments import tf_py_environment
from tf_agents.policies.tf_policy import TFPolicy

from environment.environment_wrapper import Environment_Wrapper

import tensorflow as tf
import tf_agents


class Random_Policy(TFPolicy):
    def __init__(self, tf_environment: tf_py_environment.TFPyEnvironment,
                 observation_and_action_constraint_splitter=observation_and_action_constraint_splitter):
        super(Random_Policy, self).__init__(tf_environment._time_step_spec, tf_environment._action_spec,
                                            observation_and_action_constraint_splitter=observation_and_action_constraint_splitter)
        self._time_step_spec = tf_environment._time_step_spec
        self._action_spec = tf_environment._action_spec
        self._observation_and_action_constraint_splitter = observation_and_action_constraint_splitter
        self._policy_state = None
        self._policy_info = None
        self._policy_info_spec = None

    def _distribution(self, time_step, policy_state):
        return tf_agents.trajectories.policy_step.PolicyStep(
            action=tf.random.uniform(
                (1, 1), minval=0, maxval=3, dtype=tf.int32),
            state=policy_state,
            info=self._policy_info
        )

    def _get_initial_state(self, batch_size):
        return ()

    def _action(self, time_step, policy_state, seed):
        if self.observation_and_action_constraint_splitter is not None:
            observation, mask = self.observation_and_action_constraint_splitter(
                time_step.observation)
            np_mask = mask.numpy()[0]
            possible_actions = np.where(np_mask)[0]
            selected_action = np.random.choice(possible_actions)
            selected_action = tf.convert_to_tensor(
                [selected_action], dtype=tf.int32)
            action = tf_agents.trajectories.policy_step.PolicyStep(
                action=selected_action,
                state=()
            )
        else:
            action = tf_agents.trajectories.policy_step.PolicyStep(
                action=tf.random.uniform(
                    (1, 1), minval=self._action_spec.minimum, maxval=self._action_spec.maximum, dtype=tf.int32),
                state=()
            )
        return action
