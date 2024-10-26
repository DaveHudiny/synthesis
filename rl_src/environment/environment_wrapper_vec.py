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
OBSERVATION_SIZE = 0  # Constant for valuation encoding
MAXIMUM_SIZE = 6  # Constant for reward shaping

import time

import logging

import prerequisites.VecStorm.vec_storm as vec_storm
from prerequisites.VecStorm.vec_storm.storm_vec_env import ResetInfo, StepInfo


def pad_labels(label):
            current_length = tf.shape(label)[0]
            if current_length < 1:
                return tf.pad(label, [[0, 0]], constant_values="no_label")
            else:
                return label
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

import os

def generate_reward_selection_function(rewards, labels):
    keys = rewards.keys()
    keys = list(keys)
    reward = rewards[keys[-1]]
    return reward


class Environment_Wrapper_Vec(py_environment.PyEnvironment):
    """The most important class in this project. It wraps the Stormpy simulator and provides the interface for the RL agent.
    """

    def __init__(self, stormpy_model: storage.SparsePomdp, args: ArgsEmulator, q_values_table : list[list] = None, num_envs : int = 1):
        """Initializes the environment wrapper.
        
        Args:
            stormpy_model: The Storm model to be used.
            args: The arguments from the command line or ArgsSimulator.
        """
        super(Environment_Wrapper_Vec, self).__init__()
        self.num_envs = num_envs
        self.stormpy_model = stormpy_model
        self.simulator = simulator.create_simulator(self.stormpy_model)
        labeling = stormpy_model.labeling.get_labels()
        self.special_labels = np.array(["(((sched = 0) & (t = (8 - 1))) & (k = (20 - 1)))", "goal", "done", "((x = 2) & (y = 0))",
                               "((x = (10 - 1)) & (y = (10 - 1)))"])
        intersection_labels = [label for label in labeling if label in self.special_labels]
        if not os.path.exists("simulator.pkl") or True:
            self.vectorized_simulator = vec_storm.StormVecEnv(stormpy_model, get_scalarized_reward=generate_reward_selection_function, 
                                                              num_envs=num_envs, max_steps=args.max_steps, metalabels={"goals": intersection_labels})
            with open("simulator.pkl", "wb") as f:
                import pickle as pkl
                pkl.dump(self.vectorized_simulator, f)
        else:
            with open("simulator.pkl", "rb") as f:
                import pickle as pkl
                self.vectorized_simulator = pkl.load(f)
        self.labels_mask = list([])
        self.args = args
        self.nr_obs = self.stormpy_model.nr_observations
        self.encoding_method = args.encoding_method
        self.goal_value = tf.constant(args.evaluation_goal, dtype=tf.float32)
        self.antigoal_value = tf.constant(args.evaluation_antigoal,
                                          dtype=tf.float32)
        self.goal_values_vector = tf.constant([args.evaluation_goal] * self.num_envs, dtype=tf.float32)
        self.antigoal_values_vector = tf.constant([args.evaluation_antigoal] * self.num_envs, dtype=tf.float32)
        self.discount=tf.convert_to_tensor([args.discount_factor] * self.num_envs, dtype=tf.float32)
        self.reward = tf.constant(0.0, dtype=tf.float32)
        if len(list(stormpy_model.reward_models.keys())) == 0:
            self.reward_multiplier = -1.0
        elif list(stormpy_model.reward_models.keys())[-1] in "rewards":
            self.reward_multiplier = 1.0 
        else: # If 1.0, rewards are positive, if -1.0, rewards are negative
            self.reward_multiplier = -1.0
        self._finished = False
        self._num_steps = 0
        self._current_time_step = None
        self._max_steps = args.max_steps
        self.set_action_labeling()
        self.create_specifications()
        self.action_convertor = self._convert_action

        self.last_action = 0
        self.visited_states = []
        self.empty_reward = False
        
        self.special_labels_tf = tf.constant(self.special_labels, dtype=tf.string)
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
        self.cumulative_num_steps = 0

        self.default_step_types = tf.constant([ts.StepType.MID] * self.num_envs, dtype=tf.int32)
        self.terminated_step_types = tf.constant([ts.StepType.LAST] * self.num_envs, dtype=tf.int32)

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

    def set_action_labeling(self):
        """Computes the keywords for the actions and stores them to self.act_to_keywords and other dictionaries."""
        self.action_keywords = self.vectorized_simulator.get_action_labels()
        self.action_indices = {label: i for i, label in enumerate(self.action_keywords)}
        self.nr_actions = len(self.action_keywords)
        self.act_to_keywords = dict([[self.action_indices[i], i]
                                     for i in self.action_indices])

    def create_observation_spec(self) -> tensor_spec:
        """Creates the observation spec based on the encoding method."""
        if self.encoding_method == "Valuations":
            try:
                json_example = self.stormpy_model.observation_valuations.get_json(0)
                parse_data = json.loads(str(json_example))
                observation_spec = tensor_spec.TensorSpec(shape=(
                    len(parse_data) + OBSERVATION_SIZE,), dtype=tf.float32, name="observation"),
            except:
                logging.error("Valuation encoding not possible, currently not compatible.")
                exit(0)
        else:
            raise ValueError("Encoding method currently not implemented")
        return observation_spec[0]

    def create_specifications(self):
        """Creates the specifications for the environment. Important for TF-Agents."""
        self._possible_observations = np.unique(
            self.stormpy_model.observations)
        observation_spec = self.create_observation_spec()
        integer_information = tensor_spec.TensorSpec(
            shape=(1,), dtype=tf.int32, name="integer_information")
        mask_spec = tensor_spec.TensorSpec(
            shape=(self.nr_actions,), dtype=tf.bool, name="mask")
        self._observation_spec = {
            "observation": observation_spec, "mask": mask_spec, "integer": integer_information}

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
            shape=(len(self.action_keywords),),
            dtype=tf.int32,
            minimum=0,
            maximum=len(self.action_keywords) - 1,
            name="action"
        )

    def output_spec(self) -> tensor_spec:
        return self._output_spec

    def time_step_spec(self) -> ts.TimeStep:
        return self._time_step_spec
        
    def _restart_simulator(self) -> tuple[list, list, list]:
        observations, allowed_actions, metalabels = self.vectorized_simulator.reset()
        return observations.tolist(), allowed_actions.tolist(), metalabels.tolist()

    def _reset(self) -> ts.TimeStep:
        """Resets the environment. Important for TF-Agents, since we have to restart environment many times."""
        self._finished = False
        self._num_steps = 0
        self.last_observation, self.allowed_actions, self.labels_mask = self._restart_simulator()
        # self.integer_observations = self.vectorized_simulator.observations # TODO: implement it with proposed vectorized simulator
        self.virtual_reward = tf.zeros((self.num_envs,), dtype=tf.float32)
        self.dones = np.array(len(self.last_observation) * [False])

        self.reward = tf.constant(np.array(len(self.last_observation) * [0.0]), dtype=tf.float32)
        observation_tensor = {"observation": tf.constant(self.last_observation, tf.float32), 
                              "mask": tf.constant(self.allowed_actions, tf.bool), 
                              "integer": tf.constant(tf.ones((len(self.last_observation),1), dtype=tf.int32), dtype=tf.int32)}
        self._current_time_step = ts.TimeStep(
            observation=observation_tensor, 
            reward=self.reward_multiplier * self.reward, 
            discount=self.discount, 
            step_type=tf.convert_to_tensor([ts.StepType.MID] * self.num_envs, dtype=tf.int32))
        return self._current_time_step

    def _convert_action(self, action) -> int:
        """Converts the action from the RL agent to the action used by the Vectorized simulator."""
        return action
    
    def is_goal_state(self, labels) -> bool:
        """Checks if the current state is a goal state."""
        import re
        
        combined_pattern = '|'.join([re.escape(special_label) for special_label in self.special_labels])
        goal_state_mask = self.dones & tf.math.reduce_any(
            tf.strings.regex_full_match(labels, combined_pattern),
            axis=-1
        )

        return goal_state_mask

    def evaluate_simulator(self) -> ts.TimeStep:
        """Evaluates the simulator and returns the current time step. Primarily used to determine, whether the state is the last one or not."""
        self.flag_goal = tf.zeros((self.num_envs,), dtype=tf.bool) 
        labels_mask = tf.convert_to_tensor(self.labels_mask, dtype=tf.bool)
        labels_mask = tf.reshape(labels_mask, (self.num_envs,))
        self.default_rewards = tf.constant(self.reward, dtype=tf.float32)
        antigoal_values_vector = self.antigoal_values_vector + self.default_rewards
        goal_values_vector = self.goal_values_vector + self.default_rewards
        self.goal_state_mask = labels_mask & self.dones
        # num_goals = tf.reduce_sum(tf.cast(self.goal_state_mask, tf.int32))
        # print(num_goals)
        # print(self.goal_state_mask)
        self.anti_goal_state_mask = ~labels_mask & self.dones & ~self.truncated
        still_running_mask = ~self.dones
        self.reward = tf.where(
            self.goal_state_mask,
            goal_values_vector,
            tf.where(
                self.anti_goal_state_mask,
                antigoal_values_vector,
                self.default_rewards
            )
        )
        self.step_types = tf.where(
            still_running_mask,
            self.default_step_types,
            self.terminated_step_types
        )
        self._current_time_step = ts.TimeStep(
            step_type=self.step_types,
            reward=self.reward,
            discount=self.discount,
            observation=self.get_observation()
        )
        self.virtual_reward = self.reward - self.default_rewards
        return self._current_time_step

    def is_legal_action(self, action) -> bool:
        """Checks if the action is legal."""
        act_keyword = self.act_to_keywords[int(action)]
        choice_list = self.get_choice_labels()
        if act_keyword in choice_list:
            return True
        else:
            return False

    def get_max_step_finish_timestep(self):
        """Returns the time step when the maximum number of steps is reached. Uses reward shaping, if enabled."""
        if self.reward_shaping:
            distance = self.compute_square_root_distance_from_goal()
            return ts.termination(observation=self.get_observation(),
                                  reward=tf.constant((self.goal_value / distance) / MAXIMUM_SIZE, dtype=tf.float32))
        else:
            return ts.termination(observation=self.get_observation(), reward=tf.constant(self.reward, dtype=tf.float32))

    def _do_step_in_simulator(self, actions) -> StepInfo:
        """Does the step in the Stormpy simulator.
            returns:
                tuple of new TimeStep and penalty for performed action.
        """
        self._num_steps += 1
        self.last_action = actions
        
        observations, rewards, done, allowed_actions, metalabels, truncated = self.vectorized_simulator.step(actions=actions)
        self.last_observation = observations
        self.states = self.vectorized_simulator.simulator_states
        self.allowed_actions = allowed_actions
        self.labels_mask = metalabels.tolist() # List of bools for each environment goal
        self.reward = rewards
        self.dones = done
        self.truncated = truncated
    
    def _step(self, action) -> ts.TimeStep:
        """Does the step in the environment. Important for TF-Agents and the TFPyEnvironment."""
        self.cumulative_num_steps += self.num_envs
        self._do_step_in_simulator(action)
        self.reward = tf.constant(self.reward_multiplier * self.reward, dtype=tf.float32)
        evaluated_step = self.evaluate_simulator()
        return evaluated_step
    
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
        encoded_observation = self.last_observation
        mask = self.allowed_actions
        return {"observation": tf.constant(encoded_observation, dtype=tf.float32), "mask": tf.constant(mask, dtype=tf.bool), 
                "integer": tf.constant(np.ones((len(encoded_observation), 1)), dtype=tf.int32)}

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
        observation = self.last_observation
        return observation
    
    #####################################################################################
    # Random initialization of the environment, currently deprecated
    #####################################################################################

    def get_random_legal_action(self) -> np.int32:
        available_actions = self.simulator.available_actions()
        return np.random.choice(available_actions)

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
