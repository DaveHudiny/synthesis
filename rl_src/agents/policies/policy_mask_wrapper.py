from tf_agents.policies import TFPolicy
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories import StepType

import numpy as np
import tensorflow as tf

from tf_agents.distributions import shifted_categorical
from tf_agents.trajectories import policy_step

import tensorflow_probability as tfp


from tools.encoding_methods import observation_and_action_constraint_splitter

import logging

from reward_machines.predicate_automata import PredicateAutomata

logger = logging.getLogger(__name__)

class PolicyMaskWrapper(TFPolicy):
    """Wrapper for stochastic policies that allows to use observation and action constraint splitters"""

    def __init__(self, policy: TFPolicy, observation_and_action_constraint_splitter=observation_and_action_constraint_splitter, 
                 time_step_spec=None, is_greedy : bool = False, select_rand_action_probability : float = 0.0,
                 predicate_automata : PredicateAutomata = None):
        """Initializes the policy mask wrapper, which is a wrapper for stochastic policies which enables to use observation and action constraint splitters.

        Args:
            policy (TFPolicy): Policy, which should be wrapped. This policy does not use masks for action selection.
            observation_and_action_constraint_splitter (func, optional): Splits observations to pure observations and masks. 
                                                                         Defaults to observation_and_action_constraint_splitter from agents.tools.
            time_step_spec (TimeStepSpec, optional): Time Step specification with mask. Defaults to None.
            is_greedy (bool, optional): Whether the policy should be greedy or not. Defaults to False.
        """
        if predicate_automata is not None:
            self.state_spec_len = predicate_automata.get_reward_state_spec_len(predicate_based=False)
            # Combine original spec with the length of the predicate automata visited states vector as a dictionary
            policy_state_spec = policy.policy_state_spec
            if predicate_automata.predicate_based_rewards:
                self.prediate_number = len(predicate_automata.predicate_set_labels)
                policy_state_spec = {**policy_state_spec, 'satisfied_predicates': tf.TensorSpec(shape=(len(predicate_automata.predicate_set_labels), 1), dtype=tf.bool),
                                    "automata_state": tf.TensorSpec(shape=(1,), dtype=tf.int32)}
            else:
                policy_state_spec = {**policy_state_spec, 'visited_automata_states': tf.TensorSpec(shape=(self.state_spec_len, 1), dtype=tf.bool),
                                    "automata_state": tf.TensorSpec(shape=(1,), dtype=tf.int32)}
            self.predicate_automata = predicate_automata
            self.get_initial_automata_state = self.predicate_automata.get_initial_state
            self.step_automata = self.predicate_automata.step
            self.get_initial_visited_states = self._get_initially_visited_states
            if policy.info_spec == ():
                info_spec = {'current_automata_state': tf.TensorSpec(shape=(1,), dtype=tf.int32)}
            else:
                info_spec = {**policy.info_spec, 'current_automata_state': tf.TensorSpec(shape=(1,), dtype=tf.int32)}
        else:
            policy_state_spec = policy.policy_state_spec
            self.predicate_automata = None
            info_spec = policy.info_spec

        super(PolicyMaskWrapper, self).__init__(time_step_spec=time_step_spec,
                                                  action_spec=policy.action_spec,
                                                  policy_state_spec=policy_state_spec,
                                                  info_spec=info_spec,
                                                  observation_and_action_constraint_splitter=observation_and_action_constraint_splitter)

        self._policy = policy
        self._observation_and_action_constraint_splitter = observation_and_action_constraint_splitter
        self._time_step_spec = time_step_spec
        self._action_spec = policy.action_spec
        self._policy_state_spec = policy_state_spec
        self._info_spec = policy.info_spec
        self._is_greedy = is_greedy
        self._real_distribution = self._distribution
        self._select_random_action_probability = select_rand_action_probability

        self.__policy_masker = lambda logits, mask: tf.compat.v2.where(
            tf.cast(mask, tf.bool), logits, tf.constant(logits.dtype.min, dtype=logits.dtype))
        self.__policy_dummy_masker = lambda logits, mask: logits
        self.current_masker = self.__policy_dummy_masker
        

    def is_greedy(self):
        return self._is_greedy
    
    def set_greedy(self, is_greedy):
        self._is_greedy = is_greedy

    def set_policy_masker(self):
        """Set the policy masker to the default one"""
        self.current_masker = self.__policy_masker
    
    def unset_policy_masker(self):
        """Unset the policy masker to the default one"""
        self.current_masker = self.__policy_dummy_masker

    def _get_additional_initial_state(self, batch_size):
        state_number = self.get_initial_automata_state()
        state_number = tf.fill((batch_size,), state_number)
        return state_number
    
    def _get_initially_visited_states(self, batch_size):
        # Get the initial state of the automata
        nr_visited_labels = self.state_spec_len
        # Generated zero flag mask
        visited_states = tf.zeros((batch_size, nr_visited_labels), dtype=tf.bool)
        return visited_states

    def generate_curiosity_reward(self, prev_state, next_state, next_policy_step_type):
        # Get the previously_visited_states
        if self.predicate_automata.predicate_based_rewards:
            visited_states = tf.cast(prev_state["satisfied_predicates"], tf.float32)
            next_visited_states = tf.cast(next_state["satisfied_predicates"], tf.float32)
        else:
            visited_states = tf.cast(prev_state["visited_automata_states"], tf.float32)
            next_visited_states = tf.cast(next_state["visited_automata_states"], tf.float32)
        # Get the next visited states
        
        # Compare the visited states, if there is any change for a given batch number, add a reward
        diff = tf.abs(next_visited_states - visited_states)
        # Get the reward
        reward = tf.reduce_max(diff, axis=-1) * 1
        # If there is no new state, give some small penalty
        reward = tf.where(tf.equal(reward, 0.0), -0.05, reward)
        reward = tf.where( # if the next_state is initial state, set the reward to 0
            tf.not_equal(next_policy_step_type, StepType.MID), 0.0, reward
        )
        return reward

    @tf.function
    def _get_initial_state(self, batch_size):
        # print("Getting initial state", batch_size)

        policy_state = self._policy._get_initial_state(batch_size)
        if self.predicate_automata is not None:
            if self.predicate_automata.predicate_based_rewards:
                policy_state["satisfied_predicates"] = tf.zeros((batch_size, self.prediate_number), dtype=tf.bool)
            else:
                policy_state["visited_automata_states"] = self.get_initial_visited_states(batch_size)
            policy_state["automata_state"] = self.get_initial_automata_state(batch_size)
        return policy_state

    def _distribution(self, time_step, policy_state) -> PolicyStep:
        observation, mask = self._observation_and_action_constraint_splitter(
            time_step.observation)
        time_step = time_step._replace(observation=observation)
        distribution = self._policy.distribution(
            time_step, policy_state)
        return distribution

    def _action(self, time_step, policy_state, seed) -> PolicyStep:
        # observation, mask = self._observation_and_action_constraint_splitter(time_step.observation)
        # time_step = time_step._replace(observation=observation)
        # return self._policy.action(time_step, policy_state, seed)
        distribution = self._real_distribution(time_step, policy_state)

        if self._is_greedy:
            # print("Greedy action")
            logits = distribution.action.logits
            logits = self.current_masker(logits, time_step.observation["mask"])
            action = tf.argmax(logits, output_type=tf.int32, axis=-1)
        else:
            # print("Stochastic action")
            # _, mask = self._observation_and_action_constraint_splitter(time_step.observation)
            # action = self._get_action_masked(distribution, mask)
            logits = distribution.action.logits
            logits = self.current_masker(logits, time_step.observation["mask"])
            action = tf.random.categorical(distribution.action.logits, num_samples=1, dtype=tf.int32)
            action = tf.squeeze(action, axis=-1)

        if self.predicate_automata is not None:
            new_policy_state = distribution.state
            # Get the visited states
            if self.predicate_automata.predicate_based_rewards:
                visited_states = tf.cast(policy_state["satisfied_predicates"], tf.bool)
            else:
                visited_states = tf.cast(policy_state["visited_automata_states"], tf.bool)
            # Get the next visited states
            next_visited_state, satisfied_predicates = self.step_automata(policy_state["automata_state"], time_step.observation["observation"])
            new_policy_state["automata_state"] = next_visited_state

            # Create indices from next_visited_state using batch numbers
            if self.predicate_automata.predicate_based_rewards:
                next_visited_state = tf.cast(satisfied_predicates, tf.bool)
                next_visited_state = tf.reshape(next_visited_state, (tf.shape(next_visited_state)[0], -1))
                next_visited_state = tf.logical_or(next_visited_state, visited_states)
                new_policy_state["satisfied_predicates"] = tf.cast(next_visited_state, tf.bool)
            else:
                batch_size = tf.shape(next_visited_state)[0]
                batch_indices = tf.range(batch_size)
                batch_indices = tf.reshape(batch_indices, (batch_size, 1))
                batch_indices = tf.stack([batch_indices, next_visited_state], axis=-1)
                next_state_indices = tf.reshape(batch_indices, (batch_size, 2))
                # Update the visited states with newly observed states
                visited_states = tf.tensor_scatter_nd_update(visited_states, next_state_indices, tf.ones(next_state_indices.shape[0], dtype=tf.bool))

                new_policy_state["visited_automata_states"] = visited_states
            if distribution.info == ():
                info = {'current_automata_state': new_policy_state["automata_state"]}
            else:
                distribution.info["current_automata_state"] = new_policy_state["automata_state"]
                info = distribution.info
        else:
            new_policy_state = distribution.state
            info = distribution.info
        policy_step = PolicyStep(action=action, state=new_policy_state, info=info)
        return policy_step