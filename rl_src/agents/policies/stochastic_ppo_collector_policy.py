from tf_agents.policies import TFPolicy

from tf_agents.trajectories import policy_step

import tensorflow as tf

import numpy as np

class Stochastic_PPO_Collector_Policy(TFPolicy):
    def __init__(self, tf_environment, action_spec, observation_and_action_constraint_splitter = None, collector_policy: TFPolicy = None):
        """Wrapper over a PPO collector policy to work with illegal actions. Using two options -- based on logits (no custom observation_and_action_constraint_splitter)
        or on masks. Primarily legacy code.

        Args:
            tf_environment (tf_py_environment.TFPyEnvironment): The environment to be used.
            action_spec ([type]): The action spec of the environment.
            observation_and_action_constraint_splitter ([type], optional): The observation and action constraint splitter. Defaults to None.
            collector_policy (TFPolicy, optional): The collector policy to be used. Defaults to None.
        """
        super(Stochastic_PPO_Collector_Policy, self).__init__(tf_environment.time_step_spec(
        ), action_spec, observation_and_action_constraint_splitter=observation_and_action_constraint_splitter)
        self._time_step_spec = tf_environment.time_step_spec()
        self._action_spec = action_spec
        self._observation_and_action_constraint_splitter = observation_and_action_constraint_splitter
        self._collector_policy = collector_policy
        self._policy_state = self._get_initial_state(tf_environment.batch_size)
        self._policy_state_spec = self._collector_policy.policy_state_spec
        self._policy_info_spec = collector_policy.info_spec
        self._policy_step_spec = policy_step.PolicyStep(
            action=action_spec, state=self._policy_state_spec, info=self._policy_info_spec)
        
        if observation_and_action_constraint_splitter is not None:
            self._action = self._action_masked
        else:
            pass

    def _distribution(self, time_step, policy_state):
        return self._collector_policy.distribution(time_step, policy_state)

    def _get_initial_state(self, batch_size):
        return self._collector_policy.get_initial_state(batch_size)

    def _action(self, time_step, policy_state, seed):
        action = self._collector_policy.action(time_step, policy_state, seed)
        logits = self._collector_policy.distribution(
            time_step, policy_state).action.logits
        policy_stepino = policy_step.PolicyStep(
            {"action": action.action, "logits": logits}, action.state, action.info)
        return policy_stepino
    
    def _action_masked(self, time_step, policy_state, seed):
        observation, mask = self._observation_and_action_constraint_splitter(
            time_step.observation)
        np_mask = mask.numpy()[0]
        action = self._collector_policy.action(observation, policy_state, seed)
        if np_mask[int(action.action)]:
            return action
        else:
            distribution = self._collector_policy.distribution(
                observation, policy_state)
            logits = distribution.action.logits
            actions = tf.argsort(logits, direction='DESCENDING')
            for act in actions:
                if np_mask[int(act)]:
                    action = act
                    break
            policy_stepino = policy_step.PolicyStep(
                action=action, state=distribution.state, info=distribution.info)
            return policy_stepino


