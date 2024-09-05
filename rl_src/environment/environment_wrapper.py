# Description: This file contains the environment wrapper class that is used to interact with the Storm model and the RL agent.
# Author: David Hudák
# Login: xhudak03
# File: environment_wrapper.py

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from stormpy import simulator
from stormpy.storage import storage
import stormpy

from tf_agents.environments import py_environment


from tf_agents.trajectories import time_step as ts
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step_spec
from tools.encoding_methods import *
from environment.reward_shaping_models import *

from tools.args_emulator import ArgsEmulator

import json
import argparse

OBSERVATION_SIZE = 0  # Constant for valuation encoding
MAXIMUM_SIZE = 6  # Constant for reward shaping

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)




class Environment_Wrapper(py_environment.PyEnvironment):
    """The most important class in this project. It wraps the Stormpy simulator and provides the interface for the RL agent.
    """
    def __init__(self, stormpy_model: storage.SparsePomdp, args: ArgsEmulator, q_values_table : list[list] = None):
        """Initializes the environment wrapper.
        
        Args:
            stormpy_model: The Storm model to be used.
            args: The arguments from the command line or ArgsSimulator.
        """
        super(Environment_Wrapper, self).__init__()
        self.stormpy_model = stormpy_model
        self.simulator = simulator.create_simulator(self.stormpy_model)
        self.labels = list(self.simulator._report_labels())
        self.args = args
        self.nr_obs = self.stormpy_model.nr_observations
        self.encoding_method = args.encoding_method
        self.goal_value = tf.constant(args.evaluation_goal, dtype=tf.float32)
        self.antigoal_value = tf.constant(args.evaluation_antigoal,
                                          dtype=tf.float32)
        self.discount = tf.constant(args.discount_factor, dtype=tf.float32)
        self.reward = tf.constant(0.0, dtype=tf.float32)
        if len(list(stormpy_model.reward_models.keys())) == 0:
            self.reward_multiplier = -1.0
        elif list(stormpy_model.reward_models.keys())[0] in "rewards":
            self.reward_multiplier = 1.0 
        else: # If 1.0, rewards are positive, if -1.0, rewards are negative
            self.reward_multiplier = -1.0
        self._finished = False
        self._num_steps = 0
        self._current_time_step = None
        self._max_steps = args.max_steps
        self.action_filtering = args.action_filtering
        self.illegal_action_penalty = args.illegal_action_penalty
        self.randomizing_illegal_actions = args.randomizing_illegal_actions
        self.randomizing_penalty = args.randomizing_penalty
        self.reward_shaping = args.reward_shaping
        if self.reward_shaping:
            self.select_reward_shaping_function()
        else:
            self.reward_shaping_function = lambda _: self.antigoal_value / 2
        self.compute_keywords()
        self.create_specifications()
        self.action_convertor = self._convert_action

        self.last_action = 0
        self.visited_states = []
        self.empty_reward = False
        self.special_labels = ["(((sched = 0) & (t = (8 - 1))) & (k = (20 - 1)))", "goal", "done", "((x = 2) & (y = 0))",
                               "((x = (10 - 1)) & (y = (10 - 1)))"]
        # Sometimes the goal is not labeled as "goal" but as "done" or as a special label.
        self.virtual_value = tf.constant(0.0, dtype=tf.float32)
        self.normalize_simulator_rewards = self.args.normalize_simulator_rewards
        if self.normalize_simulator_rewards:
            self.normalizer = 1.0/tf.abs(self.goal_value)
        else:
            self.normalizer = tf.constant(1.0)

        self.random_start_simulator = self.args.random_start_simulator
        self.original_init_state = self.stormpy_model.initial_states
        self.q_values_table = q_values_table

    def set_random_starts_simulation(self, flag : bool = True):
        self.random_start_simulator = flag
        if not flag:
            nr_states = self.stormpy_model.nr_states
            bitvector = stormpy.BitVector(nr_states, self.original_init_state)
            self.stormpy_model.set_initial_states(bitvector)

    def set_new_qvalues_table(self, qvalues_table):
        self.q_values_table = qvalues_table
        self.q_values_ranking = None # Because we want to re-compute the ranking later

    def set_selection_pressure(self, sp : float = 1.5):
        self.selection_pressure = sp

    def create_new_environment(self):
        return Environment_Wrapper(self.stormpy_model, self.args)

    def select_reward_shaping_function(self):
        """Selects the reward shaping function based on the arguments. Experimental feature."""
        if self.args.reward_shaping_model is None:
            self.reward_shaping_function = lambda _: self.antigoal_value / 2
        elif self.args.reward_shaping_model in ["evade", "refuel"]:
            if self.args.reward_shaping_model == "evade":
                model = EvadeRewardModel
            elif self.args.reward_shaping_model == "refuel":
                model = RefuelRewardModel
            self.reward_shaping_model = model(
                self.stormpy_model, self.simulator)
            self.reward_shaping_function = self.reward_shaping_model.reward_shaping
        else:
            raise ValueError("Reward shaping model not recognized")

    def compute_keywords(self):
        """Computes the keywords for the actions and stores them to self.act_to_keywords and other dictionaries."""
        self.action_keywords = []
        for s_i in range(self.stormpy_model.nr_states):
            n_act = self.stormpy_model.get_nr_available_actions(s_i)
            for a_i in range(n_act):
                for label in self.stormpy_model.choice_labeling.get_labels_of_choice(self.stormpy_model.get_choice_index(s_i, a_i)):
                    if label not in self.action_keywords:
                        self.action_keywords.append(label)
        self.nr_actions = len(self.action_keywords)
        self.action_indices = dict(
            [[j, i] for i, j in enumerate(self.action_keywords)])
        self.act_to_keywords = dict([[self.action_indices[i], i]
                                     for i in self.action_indices])

    def create_observation_spec(self) -> tensor_spec:
        """Creates the observation spec based on the encoding method."""
        if self.encoding_method == "One-Hot":
            observation_spec = tensor_spec.TensorSpec(shape=(
                len(self._possible_observations),), dtype=tf.float32, name="observation"),
        elif self.encoding_method == "Integer":
            observation_spec = tensor_spec.TensorSpec(
                shape=tf.TensorShape((1,)), dtype=tf.float32, name="observation"),
        elif self.encoding_method == "Valuations":
            try:
                json_example = self.stormpy_model.observation_valuations.get_json(0)
                parse_data = json.loads(str(json_example))
                observation_spec = tensor_spec.TensorSpec(shape=(
                    len(parse_data) + OBSERVATION_SIZE,), dtype=tf.float32, name="observation"),
            except:
                logging.error("Valuation encoding not possible, using one-hot encoding instead.")
                observation_spec = tensor_spec.TensorSpec(shape=(
                    len(self._possible_observations),), dtype=tf.float32, name="observation"),
                self.args = "One-Hot"
        elif self.encoding_method == "Valuations++":
            try:
                json_example = self.stormpy_model.observation_valuations.get_json(0)
                parse_data = json.loads(str(json_example))
                observation_spec = tensor_spec.TensorSpec(shape=(
                    len(parse_data) + OBSERVATION_SIZE + 1,), dtype=tf.float32, name="observation"),
            except:
                logging.error("Valuation encoding not possible, using one-hot encoding instead.")
                observation_spec = tensor_spec.TensorSpec(shape=(
                    len(self._possible_observations),), dtype=tf.float32, name="observation"),
                self.args = "One-Hot"

        else:
            raise ValueError("Encoding method not recognized")
        return observation_spec[0]

    def create_specifications(self):
        """Creates the specifications for the environment. Important for TF-Agents."""
        self._possible_observations = np.unique(
            self.stormpy_model.observations)
        observation_spec = self.create_observation_spec()
        integer_information = tensor_spec.TensorSpec(
            shape=(1,), dtype=tf.int32, name="integer_information")
        if not self.action_filtering:
            mask_spec = tensor_spec.TensorSpec(
                shape=(self.nr_actions,), dtype=tf.bool, name="mask")
            self._observation_spec = {
                "observation": observation_spec, "mask": mask_spec, "integer": integer_information}
        else:
            self._observation_spec = observation_spec

        self._time_step_spec = time_step_spec(
            observation_spec=self._observation_spec,
            reward_spec=tensor_spec.TensorSpec(
                shape=(), dtype=tf.float32, name="reward"),
        )
        self._action_spec = tensor_spec.BoundedTensorSpec(
            shape=(),
            dtype=tf.int32,
            minimum=0,
            maximum=len(self.action_keywords) - 1,
            name="action"
        )
        self._output_spec = tensor_spec.BoundedTensorSpec(
            shape=(len(self.action_keywords)),
            dtype=tf.int32,
            minimum=0,
            maximum=len(self.action_keywords) - 1,
            name="action"
        )

    def output_spec(self) -> tensor_spec:
        return self._output_spec

    def time_step_spec(self) -> ts.TimeStep:
        return self._time_step_spec

    def compute_mask(self) -> tf.Tensor:
        """Computes the mask for the actions based on the current state. True means the action is legal, False means it is illegal.
           The mask is focused on idexation used by the RL agent.
        """
        choice_index = self.stormpy_model.get_choice_index(
            self.simulator._report_state(), 0)
        if len(self.stormpy_model.choice_labeling.get_labels_of_choice(choice_index)) == 0:
            mask_inds = [0]
        else:
            available_actions = self.simulator.available_actions()
            mask_inds = []
            for a_i in available_actions:
                choice_index = self.stormpy_model.get_choice_index(
                    self.simulator._report_state(), a_i)
                mask_inds.append(
                    self.action_indices[self.stormpy_model.choice_labeling.get_labels_of_choice(choice_index).pop()])
        mask = np.zeros(shape=(self.nr_actions,), dtype=bool)
        for i in mask_inds:
            mask[i] = True
        mask = tf.logical_and(
            tf.ones(shape=(1, self.nr_actions), dtype=tf.bool), mask)
        return mask

    def create_encoding(self, observation) -> tf.Tensor:
        """Creates the encoding for the observation based on the encoding method."""
        if self.encoding_method == "One-Hot":
            observation_vector = create_one_hot_encoding(
                observation, self._possible_observations)
            return tf.constant(observation_vector, dtype=tf.float32)
        elif self.encoding_method == "Integer":
            return tf.constant([observation], dtype=tf.float32)
        elif self.encoding_method == "Valuations":
            observation_vector = create_valuations_encoding(
                observation, self.stormpy_model)
            return tf.constant(observation_vector, dtype=tf.float32)
        elif self.encoding_method == "Valuations++":
            observation_vector = create_valuations_encoding_plus(
                observation, self.stormpy_model, self.simulator._report_state())
            return tf.constant(observation_vector, dtype=tf.float32)
        
    def _set_init_state(self, index : int = 0):
        nr_states = self.stormpy_model.nr_states
        indices_bitvector = stormpy.BitVector(nr_states, [index])
        self.stormpy_model.set_initial_states(indices_bitvector)
        
    def _uniformly_change_init_state(self):
        nr_states = self.stormpy_model.nr_states
        index = np.random.randint(0, nr_states)
        self._set_init_state(index)

    @tf.function
    def sort_q_values(self, q_values_table) -> tf.Tensor:
        maximums = tf.reduce_max(q_values_table, axis=-1)
        arg_sorted_qvalues = tf.argsort(maximums, direction="DESCENDING")
        rank_tensor = tf.zeros_like(maximums, dtype=tf.int32)
        rank_tensor = tf.tensor_scatter_nd_update(rank_tensor, 
                                                tf.expand_dims(arg_sorted_qvalues, axis=1), 
                                                tf.range(tf.size(maximums)))
        # logger.info("Computed q-values ranking.")
        return rank_tensor + 1


    @tf.function
    def compute_rank_based_probabilities(self, selection_pressure, arg_sorted_q_values, n) -> tf.Tensor:
        probabilities = (arg_sorted_q_values - 1) / (n - 1)
        probabilities = (2 * selection_pressure - 2) * probabilities
        probabilities = (selection_pressure - probabilities) / n
        return probabilities
    
    def _rank_selection(self, selection_pressure : float = 1.4) -> int:
        """Selection based on rank selection in genetic algorithms. See https://en.wikipedia.org/wiki/Selection_(genetic_algorithm).

        Args:
            selection_pressure (float): Rate of selection pressure. 1 means totally random, 2 means high selection pressure.
        """
        n = self.stormpy_model.nr_states
        updated = False
        if not hasattr(self, "q_values_ranking") or self.q_values_ranking is None:
            self.q_values_ranking = self.sort_q_values(self.q_values_table)
            updated = True
        if updated or (not hasattr(self, "dist")) or self.dist is None:
            probabilities = self.compute_rank_based_probabilities(selection_pressure, self.q_values_ranking, n)
            self.dist = tfp.distributions.Categorical(probs=probabilities)
        index = self.dist.sample().numpy()
        return index
    
    def _change_init_state_by_q_values_ranked(self):
        if hasattr(self, "selection_pressure"):
            index = self._rank_selection(self.selection_pressure)
        else:
            index = self._rank_selection()
        self._set_init_state(index)
        
    def _restart_simulator(self):
        if self.random_start_simulator:
            if self.q_values_table is None:
                randomly_change_init_state = self._uniformly_change_init_state
            else:
                randomly_change_init_state = self._change_init_state_by_q_values_ranked
            randomly_change_init_state()
            stepino = self.simulator.restart()
            while self.simulator.is_done():
                randomly_change_init_state()
                stepino = self.simulator.restart()
            return stepino
        else:
            return self.simulator.restart()

    def _reset(self) -> ts.TimeStep:
        """Resets the environment. Important for TF-Agents, since we have to restart environment many times."""
        self._finished = False
        self._num_steps = 0
        stepino = self._restart_simulator()
        self.labels = list(self.simulator._report_labels())
        self.virtual_value = tf.constant(0.0, dtype=tf.float32)
        observation = stepino[0]
        if stepino[1] == []:
            self.empty_reward = True
            reward = tf.constant(0, dtype=tf.float32)
        else:
            reward = tf.constant(self.reward_multiplier * stepino[1][0] * self.normalizer, dtype=tf.float32)
        if not self.action_filtering:
            mask = self.compute_mask()
            observation_vector = self.create_encoding(observation)
            observation_tensor = {
                "observation": observation_vector, "mask": tf.constant(mask[0], dtype=tf.bool), "integer": tf.constant([observation], dtype=tf.int32)}
        else:
            observation_tensor = self.create_encoding(observation)
        self._current_time_step = ts.TimeStep(
            observation=observation_tensor, 
            reward=self.reward_multiplier * reward, 
            discount=self.discount, 
            step_type=ts.StepType.FIRST)
        
        return self._current_time_step

    def _convert_action(self, action) -> int:
        """Converts the action from the RL agent to the action used by the Storm model."""
        act_keyword = self.act_to_keywords[int(action)]
        choice_list = self.get_choice_labels()
        try:
            action = choice_list.index(act_keyword)
        except: # Should not happen much, probably broken agent!
            action = 0
        return action

    def compute_square_root_distance_from_goal(self) -> float:
        """Computes the square root distance from the goal. Used for reward shaping."""
        self.simulator.set_observation_mode(
            stormpy.simulator.SimulatorObservationMode.PROGRAM_LEVEL)
        json_final = json.loads(str(self.simulator._report_state()))
        ax = json_final["dx"]
        ay = json_final["dy"]
        distance = (MAXIMUM_SIZE - ax) + (MAXIMUM_SIZE - ay)
        self.simulator.set_observation_mode(
            stormpy.simulator.SimulatorObservationMode.STATE_LEVEL)
        return distance
    
    def get_coordinates(self) -> tuple:
        """Gets the coordinates of the agent. Experimental feature used for exact reward shaping."""
        self.simulator.set_observation_mode(
            stormpy.simulator.SimulatorObservationMode.PROGRAM_LEVEL)
        json_final = json.loads(str(self.simulator._report_state()))
        ax = json_final["ax"]
        ay = json_final["ay"]
        self.simulator.set_observation_mode(
            stormpy.simulator.SimulatorObservationMode.STATE_LEVEL)
        return ax, ay
    
    def is_goal_state(self, labels) -> bool:
        """Checks if the current state is a goal state."""
        for label in labels:
            if label in self.special_labels:
                return True
        return False

    def evaluate_simulator(self) -> ts.TimeStep:
        """Evaluates the simulator and returns the current time step. Primarily used to determine, whether the state is the last one or not."""
        self.labels = list(self.simulator._report_labels())
        self.flag_goal = False
        if self._num_steps >= self._max_steps:
            self._finished = True
            self._current_time_step = self.get_max_step_finish_timestep()
        elif not self.simulator.is_done():
            self._current_time_step = ts.transition(
                observation=self.get_observation(), reward=self.reward, discount=self.discount)
        # elif self.simulator.is_done() and ("goal" in labels or "done" in labels or "((x = 2) & (y = 0))" in labels or labels == self.special_labels):
        elif self.simulator.is_done() and self.is_goal_state(self.labels):
            logging.info("Goal reached!")
            self._finished = True
            self.flag_goal = True
            self.virtual_value = self.goal_value
            self._current_time_step = ts.termination(
                observation=self.get_observation(), reward=self.goal_value + self.reward)
        elif self.simulator.is_done() and "traps" in self.labels:
            logging.info("Trapped!")
            self._finished = True
            self.virtual_value = self.antigoal_value
            self._current_time_step = ts.termination(
                observation=self.get_observation(), reward=self.antigoal_value + self.reward)
        else:  # Ended, but not in goal state :/
            logging.info(f"Ended, but not in a goal state: {self.labels}")
            self._finished = True
            self.virtual_value = self.antigoal_value
            self._current_time_step = ts.termination(
                observation=self.get_observation(), reward=self.antigoal_value + self.reward)
        return self._current_time_step

    def is_legal_action(self, action) -> bool:
        """Checks if the action is legal."""
        act_keyword = self.act_to_keywords[int(action)]
        choice_list = self.get_choice_labels()
        if act_keyword in choice_list:
            return True
        else:
            return False

    def get_random_legal_action(self) -> np.int32:
        available_actions = self.simulator.available_actions()
        return np.random.choice(available_actions)

    def get_max_step_finish_timestep(self):
        """Returns the time step when the maximum number of steps is reached. Uses reward shaping, if enabled."""
        if self.reward_shaping:
            distance = self.compute_square_root_distance_from_goal()
            return ts.termination(observation=self.get_observation(),
                                  reward=tf.constant((self.goal_value / distance) / MAXIMUM_SIZE, dtype=tf.float32))
        else:
            return ts.termination(observation=self.get_observation(), reward=tf.constant(self.reward, dtype=tf.float32))

    def _do_step_in_simulator(self, action) -> tuple[ts.TimeStep, float]:
        """Does the step in the Stormpy simulator.
        
            returns:
                tuple of new TimeStep and penalty for performed action"""
        penalty = 0.0
        self._num_steps += 1
        self.last_action = action
        if not self.action_filtering:
            action = self.action_convertor(action)
            stepino = self.simulator.step(int(action))
        else:
            if not self.is_legal_action(action):
                if self.randomizing_illegal_actions:
                    action = self.get_random_legal_action()
                    penalty = self.randomizing_penalty
                else:
                    self._current_time_step = ts.transition(
                        observation=self.get_observation(), reward=tf.constant(self.illegal_action_penalty, dtype=tf.float32), discount=self.discount)
                    return self._current_time_step
            else:
                action = self._convert_action(action)
            stepino = self.simulator.step(int(action))
        return stepino, penalty

    def _step(self, action) -> ts.TimeStep:
        """Does the step in the environment. Important for TF-Agents and the TFPyEnvironment."""
        self.virtual_value = tf.constant(0.0, dtype=tf.float32)
        if self._finished:
            self._finished = False
            return self._reset()
        stepino, penalty = self._do_step_in_simulator(action)
        simulator_reward = stepino[1][-1] if not self.empty_reward else 1.0
        if "traps" in list(self.simulator._report_labels()):
            self.reward = self.antigoal_value
        else:
            self.reward = tf.constant(
                self.reward_multiplier * simulator_reward + penalty, dtype=tf.float32)
        evaluated_step = self.evaluate_simulator()
        normalized_step = self.normalize_reward_in_time_step(evaluated_step)
        # print(normalized_step)
        return normalized_step
    
    def normalize_reward_in_time_step(self, time_step : ts.TimeStep):
        new_reward = time_step.reward * self.normalizer
        new_time_step = ts.TimeStep(
            step_type=time_step.step_type,
            reward=new_reward,
            discount=time_step.discount,
            observation=time_step.observation
        )
        return new_time_step

    def current_time_step(self) -> ts.TimeStep:
        return self._current_time_step

    def observation_spec(self) -> ts.tensor_spec:
        return self._observation_spec

    def action_spec(self) -> ts.tensor_spec:
        return self._action_spec

    def get_observation(self) -> dict[str: tf.Tensor]:
        observation = self.simulator._report_observation()
        if not self.action_filtering:
            mask = self.compute_mask()
            return {"observation": self.create_encoding(observation), "mask": tf.constant(mask[0], dtype=tf.bool), 
                    "integer": tf.constant([observation], dtype=tf.int32)}
        else:
            return self.create_encoding(observation)

    def get_choice_labels(self) -> list[str]:
        """Converts the current legal actions to the keywords used by the Storm model."""
        labels = []
        for action_index in range(self.simulator.nr_available_actions()):
            report_state = self.simulator._report_state()
            choice_index = self.stormpy_model.get_choice_index(
                report_state, action_index)
            labels_of_choice = self.stormpy_model.choice_labeling.get_labels_of_choice(
                choice_index)
            label = labels_of_choice.pop()
            labels.append(label)
        return labels

    def get_simulator_observation(self) -> int:
        observation = self.simulator._report_observation()
        return observation
