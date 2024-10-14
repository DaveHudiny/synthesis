# Description: Controller for the simulation of the SAYNT algorithm in the Storm simulator.
# Author: David Hudák
# Login: ihudak


import paynt.quotient.storm_pomdp_control as Storm_POMDP_Control
import paynt.quotient.pomdp as POMDP
from paynt.rl_extension.saynt_rl_tools.simulator import init_simulator

import stormpy.storage as Storage
import stormpy
import re


import tensorflow as tf
from tf_agents.trajectories import StepType

from paynt.quotient.fsc import FSC

from paynt.rl_extension.saynt_controller.saynt_step import SAYNT_Step
from paynt.rl_extension.saynt_controller.saynt_modes import SAYNT_Modes

import logging

import numpy as np

logger = logging.getLogger(__name__)

class SAYNT_Simulation_Controller:
    """Class for controller applicable in Storm simulator (or similar).
    """
    MODES = ["BELIEF", "Cutoff_FSC", "Scheduler"]

    def __init__(self, storm_control: Storm_POMDP_Control.StormPOMDPControl, quotient: POMDP.PomdpQuotient,
                 tf_action_labels: list = None, max_step_limit: int = 800, goal_reward: float = 100,
                 fsc: FSC = None):
        """Initialization of the controller.
        Args:
            storm_control: Result of the SAYNT algorithm.
            quotient: Important structure containing various information about the model etc.
            tf_action_labels: List of action labels. Index of action label correspond to output node.
        """
        self.storm_control = storm_control
        self.storm_control_result = storm_control.latest_storm_result
        self.quotient = quotient
        self.current_state = None
        self.current_mode = SAYNT_Modes.BELIEF
        self.tf_action_labels = tf_action_labels
        self.induced_mc_nr_states = self.storm_control_result.induced_mc_from_scheduler.nr_states
        self.max_step_limit = max_step_limit
        self.steps_performed = 0
        self.goal_reward = goal_reward
        self.simulator = None  # Simulator is initialized in cutoff states
        self.fsc = fsc

        self.num_observations = quotient.pomdp.nr_observations
        self.get_choice_label = self.storm_control_result.induced_mc_from_scheduler.choice_labeling.get_labels_of_choice
        self.special_labels = [
            "(((sched = 0) & (t = (8 - 1))) & (k = (20 - 1)))", "goal", "done", "((x = 2) & (y = 0))"]

    def is_goal_state(self, labels):
        """Checks if the current state is a goal state."""
        for label in labels:
            if label in self.special_labels:
                return True
        return False

    def get_next_step(self, prev_step: SAYNT_Step) -> SAYNT_Step:
        """Get the next action.
        Args:
            prev_step: Previous SAYNT_Step.
        Returns:
            SAYNT_Step: Next step.
        """
        self.current_state = prev_step
        self.current_mode = prev_step.new_mode
        self.steps_performed += 1
        if self.current_mode == SAYNT_Modes.BELIEF:
            return self.get_next_step_belief(prev_step)
        elif self.current_mode == SAYNT_Modes.CUTOFF_FSC:
            return self.get_next_step_cutoff_fsc(prev_step)
        elif self.current_mode == SAYNT_Modes.CUTOFF_SCHEDULER:
            return self.get_next_step_cutoff_scheduler(prev_step)
        else:
            raise ValueError("Unknown mode")

    def get_observations_and_action_from_labels(self, state: Storage.SparseModelState):
        observations = []
        actions = []
        for label in state.labels:  # Is there really needed a for loop?
            if '[' in label:
                observation = self.quotient.observation_labels.index(label)
            elif 'obs_' in label:
                _, observation = label.split('_')
            else:
                continue
            choice_label = list(self.get_choice_label(state.id))[0]
            try:
                index = self.tf_action_labels.index(choice_label)
            except ValueError:
                index = -1
            observations.append(observation)
            actions.append(index)
        return observations[0], actions[0]

    def update_state(self, state: Storage.SparseModelState):
        """Function samples new state from transition matrix of induced MC given current state."""
        probs = self.storm_control.latest_storm_result.induced_mc_from_scheduler.transition_matrix[
            state.id]
        prob_row = np.zeros((self.induced_mc_nr_states))
        for prob_key in probs:
            prob_row[prob_key.column] = prob_key.value()
        logits = tf.math.log([prob_row])
        sample = tf.random.categorical(logits, num_samples=1)
        index = tf.squeeze(sample).numpy()
        new_state = self.storm_control.latest_storm_result.induced_mc_from_scheduler.states[
            index]
        return new_state

    def get_new_mode(self, state: Storage.SparseModelState):
        new_mode = None
        if "cutoff" not in state.labels and 'clipping' not in state.labels:
            new_mode = SAYNT_Modes.BELIEF
        else:
            if "finite_mem" in state.labels:
                new_mode = SAYNT_Modes.CUTOFF_FSC
            else:
                new_mode = SAYNT_Modes.CUTOFF_SCHEDULER
        return new_mode

    def get_tf_step_type(self, state):
        is_last = self.storm_control_result.induced_mc_from_scheduler.is_sink_state(
            state.id)
        if is_last or self.steps_performed > self.max_step_limit:
            return StepType.LAST
        else:
            return StepType.MID

    def get_reward(self, state, step_type):
        if step_type == StepType.LAST:
            if "target" in state.labels:
                return self.goal_reward
            else:
                return -self.goal_reward
        else:
            return -1  # Currently only a simple reward model

    def get_next_step_belief(self, prev_step: SAYNT_Step) -> SAYNT_Step:
        """Get the next step in belief mode.
        Returns:
            SAYNT_Step: Next step.
        """
        state = self.update_state(prev_step.state)
        observation, action = self.get_observations_and_action_from_labels(
            prev_step.state)
        new_mode = self.get_new_mode(state)
        tf_step_type = self.get_tf_step_type(state)
        reward = self.get_reward(state, tf_step_type)
        new_step = SAYNT_Step(action, observation, state, new_mode,
                              tf_step_type, reward, integer_observation=observation)
        return new_step

    def get_choice_from_scheduler(self, scheduler, state):
        choice = scheduler.get_choice(state)

        if choice.deterministic:
            selected_choice = choice.get_deterministic_choice()
        else:
            matches = re.findall(
                r"\[(\d+\.\d+): (\d+)\]", choice.get_choice().__str__())
            parsed_dict = {int(action): float(probability)
                           for probability, action in matches}
            normalized_probs = list(parsed_dict.values()) / \
                np.sum(list(parsed_dict.values()))
            selected_choice = np.random.choice(
                list(parsed_dict.keys()), p=normalized_probs)
        return int(selected_choice)

    def get_next_step_cutoff_scheduler(self, prev_step: SAYNT_Step) -> SAYNT_Step:
        """Get the next step in cutoff scheduler mode.
        Returns:
            SAYNT_Step: Next step.
        """
        if self.simulator is None:
            self.simulator = init_simulator(
                self.quotient.pomdp, prev_step.integer_observation)
        if 'sched_' in list(self.get_choice_label(prev_step.state.id))[0]:
            _, scheduler_index = list(self.get_choice_label(
                prev_step.state.id))[0].split('_')
        else:
            raise "Missing scheduler for scheduler branch :("
        scheduler = self.storm_control_result.cutoff_schedulers[int(
            scheduler_index)]
        # Not against the rules of partial observability, as all states given some observation emits same actions for a scheduler.
        state = self.simulator._report_state()
        choice = self.get_choice_from_scheduler(scheduler, state)
        choice_labels = self.get_choice_labels()
        choice_label = choice_labels[choice]
        action = self.tf_action_labels.index(choice_label)
        observation, rewards, _ = self.simulator.step(choice)
        reward = self.get_simulator_reward(rewards)
        tf_step_type = self._get_simulation_step_type()
        new_step = SAYNT_Step(action=action, observation=observation, state=prev_step.state,
                              new_mode=SAYNT_Modes.CUTOFF_SCHEDULER, tf_step_type=tf_step_type,
                              reward=reward, integer_observation=observation)

        return new_step

    def get_simulator_reward(self, sim_step_rewards):
        labels = list(self.simulator._report_labels())
        if self.is_goal_state(labels):
            reward = self.goal_reward - sim_step_rewards[-1]
        else:
            reward = - sim_step_rewards[-1]
        return reward

    def convert_fsc_action_to_tf_action(self, action_number):
        keyword = self.fsc.action_labels[action_number]
        if keyword == "__no_label__":
            return tf.constant(-1, dtype=tf.int32)
        tf_action_number = tf.argmax(
            tf.cast(tf.equal(self.tf_action_labels, keyword), tf.int32), output_type=tf.int32)
        return tf_action_number, keyword

    def _convert_action(self, act_keyword):
        """Converts the action from the RL agent to the action used by the Storm model."""
        choice_list = self.get_choice_labels()
        try:
            action = choice_list.index(act_keyword)
        except:  # Should not happen much, probably broken agent!
            action = 0
        return action

    def get_choice_labels(self):
        """Converts the current legal actions to the keywords used by the Storm model."""
        labels = []
        for action_index in range(self.simulator.nr_available_actions()):
            report_state = self.simulator._report_state()
            choice_index = self.quotient.pomdp.get_choice_index(
                report_state, action_index)
            labels_of_choice = self.quotient.pomdp.choice_labeling.get_labels_of_choice(
                choice_index)
            label = labels_of_choice.pop()
            labels.append(label)
        return labels

    def _get_simulation_step_type(self):
        if self.simulator.is_done() or self.steps_performed > self.max_step_limit:
            tf_step_type = StepType.LAST
        else:
            tf_step_type = StepType.MID
        return tf_step_type

    def get_next_step_cutoff_fsc(self, prev_step: SAYNT_Step) -> SAYNT_Step:
        """Get the next step in cutoff FSC mode.
        Returns:
            SAYNT_Step: Next step.
        """
        if self.simulator is None:
            self.simulator = init_simulator(
                self.quotient.pomdp, observation=prev_step.integer_observation)
        action = self.fsc.action_function[prev_step.fsc_memory][prev_step.observation]
        new_memory = self.fsc.update_function[prev_step.fsc_memory][prev_step.observation]
        tf_action, action_label = self.convert_fsc_action_to_tf_action(action)
        action = self._convert_action(action_label)
        sim_step = self.simulator.step(action)
        observation, rewards, _ = sim_step

        reward = self.get_simulator_reward(rewards)
        tf_step_type = self._get_simulation_step_type()
        new_saynt_step = SAYNT_Step(tf_action, observation, prev_step.state, new_mode=SAYNT_Modes.CUTOFF_FSC,
                                    tf_step_type=tf_step_type, reward=reward, fsc_memory=new_memory,
                                    integer_observation=observation)

        return new_saynt_step

    def reset(self) -> SAYNT_Step:
        """Resets the simulation with setting current state to initial state.
        Returns:
            SAYNT_Step: Initial state of induced MC.
        """
        if self.simulator is not None:
            self.simulator = None
        self.steps_performed = 0
        init_states = self.storm_control_result.induced_mc_from_scheduler.initial_states
        n = len(init_states)
        index = tf.random.uniform(shape=[], minval=0, maxval=n, dtype=tf.int32)
        sample = tf.gather(init_states, index)
        state = self.storm_control_result.induced_mc_from_scheduler.states[sample]
        observation, action = self.get_observations_and_action_from_labels(
            state)
        mode = self.get_new_mode(state)
        saynt_step = SAYNT_Step(action, observation,
                                state, mode, StepType.FIRST)
        return saynt_step