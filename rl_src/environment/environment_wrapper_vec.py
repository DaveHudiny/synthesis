# Description: This file contains the environment wrapper class that is used to interact with the Storm model and the RL agent.
# Author: David Hudák
# Login: xhudak03
# File: environment_wrapper.py

import os
from vec_storm.storm_vec_env import StepInfo
import logging
import numpy as np
import tensorflow as tf

from stormpy import simulator
from stormpy.storage import storage

from environment import py_environment


from tf_agents.trajectories import time_step as ts
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step_spec
from tools.encoding_methods import *

from tools.args_emulator import ArgsEmulator
from environment.vectorized_sim_initializer import SimulatorInitializer


import json

import jax


OBSERVATION_SIZE = 0  # Constant for valuation encoding
MAXIMUM_SIZE = 6  # Constant for reward shaping


def pad_labels(label):
    current_length = tf.shape(label)[0]
    if current_length < 1:
        return tf.pad(label, [[0, 0]], constant_values="no_label")
    else:
        return label


logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

logging.getLogger("jax").setLevel(logging.ERROR)
os.environ["JAX_LOG_LEVEL"] = "ERROR"


def generate_reward_selection_function(rewards, labels):
    last_reward = labels[-1]
    return rewards[last_reward]


class Environment_Wrapper_Vec(py_environment.PyEnvironment):
    """The most important class in this project. It wraps the Stormpy simulator and provides the interface for the RL agent.
    """

    def __init__(self, stormpy_model: storage.SparsePomdp, args: ArgsEmulator, q_values_table: list[list] = None, num_envs: int = 1):
        """Initializes the environment wrapper.

        Args:
            stormpy_model: The Storm model to be used.
            args: The arguments from the command line or ArgsSimulator.
        """
        self.args = args
        super(Environment_Wrapper_Vec, self).__init__()
        # self.batched = True
        # self.batch_size = num_envs
        self.num_envs = num_envs
        self.stormpy_model = stormpy_model
        self.state_to_observation_map = tf.constant(stormpy_model.observations)
        # Special labels representing the typical labels of goal states. If the model has different label for goal state, we should add it here.
        # TODO: What if we want to minimize the probability of reaching some state or we want to maximize the probability of reaching some other state?
        self.special_labels = np.array(["(((sched = 0) & (t = (8 - 1))) & (k = (20 - 1)))", "goal", "done", "((x = 2) & (y = 0))",
                                        "((x = (10 - 1)) & (y = (10 - 1)))"])

        # Initialization of the vectorized simulator.
        labeling = stormpy_model.labeling.get_labels()
        intersection_labels = [
            label for label in labeling if label in self.special_labels]
        metalabels = {"goals": intersection_labels}

        self.vectorized_simulator = SimulatorInitializer.load_and_store_simulator(
            stormpy_model=stormpy_model, get_scalarized_reward=generate_reward_selection_function, num_envs=num_envs,
            max_steps=args.max_steps, metalabels=metalabels, model_path=args.prism_model)
        self.vectorized_simulator.set_num_envs(num_envs)
        self.vectorized_simulator.reset()

        # Default labels mask for the environment given that the number of metalabels is 1 ("goals").
        self.labels_mask = list([False] * self.num_envs)
        self.encoding_method = args.encoding_method

        # Initialization of the goal and antigoal values for the evaluation of the environment. These goals represent virtual values for achieving the goal or other states.
        self.goal_values_vector = tf.constant(
            [args.evaluation_goal] * self.num_envs, dtype=tf.float32)
        self.antigoal_values_vector = tf.constant(
            [args.evaluation_antigoal] * self.num_envs, dtype=tf.float32)

        # Initialization of the penalty for illegal actions.
        self.flag_penalty = args.flag_illegal_action_penalty
        self.illegal_action_penalty = tf.constant(
            [self.args.illegal_action_penalty_per_step] * self.num_envs, dtype=tf.float32)

        # Initialization of the discount factor for the environment.
        self.discount = tf.convert_to_tensor(
            [args.discount_factor] * self.num_envs, dtype=tf.float32)

        # Initialization of the rewards before simulation.
        self.reward = tf.constant(0.0, dtype=tf.float32)

        # Initialization of the reward multiplier for different tasks.
        if len(list(stormpy_model.reward_models.keys())) == 0:
            self.reward_multiplier = -1.0
        elif list(stormpy_model.reward_models.keys())[-1] in "rewards":
            self.reward_multiplier = 1.0
        # If 1.0, rewards are positive, if -1.0, rewards are negative (penalties -- we try to minimize them)
        else:
            self.reward_multiplier = -1.0

        self._current_time_step = None

        # Initialization of the TF Agents specifications
        self.set_action_labeling()

        # Initialization of the observation spec.
        self.model_memory_size = args.model_memory_size # If > 0, the model provides agent with current memory and allows agent to update it.
        self.create_specifications()

        # Normalization of the rewards. Useless for PPO with its own normalization.
        self.goal_value = tf.constant(args.evaluation_goal, dtype=tf.float32)
        self.normalize_simulator_rewards = self.args.normalize_simulator_rewards
        if self.normalize_simulator_rewards:
            self.normalizer = 1.0/tf.abs(self.goal_value)
        else:
            self.normalizer = tf.constant(1.0)

        # Information about the environment.
        self.random_start_simulator = self.args.random_start_simulator

        # Statistic for debugging purposes.
        self.cumulative_num_steps = 0

        # Step types used in the environment evaluation.
        self.init_step_types = tf.constant(
            [ts.StepType.FIRST] * self.num_envs, dtype=tf.int32)
        self.default_step_types = tf.constant(
            [ts.StepType.MID] * self.num_envs, dtype=tf.int32)
        self.terminated_step_types = tf.constant(
            [ts.StepType.LAST] * self.num_envs, dtype=tf.int32)

        # Add reward shaping
        self.reward_shaper_function = lambda observation, _: tf.zeros(
            (observation.shape[0],), dtype=tf.float32)

    def set_reward_shaper(self, reward_shaper_function):
        self.reward_shaper_function = reward_shaper_function

    def unset_reward_shaper(self):
        self.reward_shaper_function = lambda observation, _: tf.zeros(
            (observation.shape[0],), dtype=tf.float32)

    def set_action_labeling(self):
        """Computes the keywords for the actions and stores them to self.act_to_keywords and other dictionaries."""
        self.action_keywords = self.vectorized_simulator.get_action_labels()
        self.action_indices = {label: i for i,
                               label in enumerate(self.action_keywords)}
        self.nr_actions = len(self.action_keywords)
        self.act_to_keywords = dict([[self.action_indices[i], i]
                                     for i in self.action_indices])

    def set_random_starts_simulation(self, randomized_bool: bool = True):
        self.random_start_simulator = randomized_bool
        if randomized_bool:
            self.vectorized_simulator.enable_random_init()
        else:
            self.vectorized_simulator.disable_random_init()

    def create_observation_spec(self) -> tensor_spec:
        """Creates the observation spec based on the encoding method."""
        if self.model_memory_size > 0:
            action_size_const = 1
        else:
            action_size_const = 0
        if self.encoding_method == "Valuations":
            try:
                json_example = self.stormpy_model.observation_valuations.get_json(
                    0)
                parse_data = json.loads(str(json_example))
                observation_spec = tensor_spec.TensorSpec(shape=(
                    len(parse_data) + OBSERVATION_SIZE + action_size_const,), dtype=tf.float32, name="observation"),
            except:
                logging.error(
                    "Valuation encoding not possible, currently not compatible. Probably model issue.")
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
        if self.model_memory_size > 0:
            self._action_spec = {
                "simulator_action": tensor_spec.BoundedTensorSpec(
                    shape=(),
                    dtype=tf.int32,
                    minimum=0,
                    maximum=len(self.action_keywords) - 1,
                    name="action"
                ),
                "memory_update": tensor_spec.BoundedTensorSpec(
                    shape=(),
                    dtype=tf.int32,
                    minimum=0,
                    maximum=len(self.action_keywords) - 1,
                    name="memory_update"
                )
            }
        else:
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

    def set_num_envs(self, num_envs: int):
        self.num_envs = num_envs
        self.vectorized_simulator.set_num_envs(num_envs)
        self._reset()

    def _reset(self) -> ts.TimeStep:
        """Resets the environment. Important for TF-Agents, since we have to restart environment many times."""
        logger.info("Resetting the environment.")
        self.last_observation, self.allowed_actions, self.labels_mask = self._restart_simulator()
        # self.integer_observations = self.vectorized_simulator.observations # TODO: implement it with proposed vectorized simulator
        self.virtual_reward = tf.zeros((self.num_envs,), dtype=tf.float32)
        self.dones = np.array(len(self.last_observation) * [False])

        self.reward = tf.constant(
            np.array(len(self.last_observation) * [0.0]), dtype=tf.float32)
        hidden_state = self.vectorized_simulator.simulator_states
        self.integers = tf.reshape(
            tf.gather(self.state_to_observation_map, hidden_state.vertices),
            (self.num_envs, 1)
        )
        if self.model_memory_size > 0:
            default_memory = tf.zeros((self.num_envs, 1), dtype=tf.float32)
            self.memory_update = default_memory
            observation_tensor = {"observation": tf.concat([tf.constant(self.last_observation, dtype=tf.float32), default_memory], axis=1),
                                  "mask": tf.constant(self.allowed_actions, tf.bool),
                                  "integer": self.integers}
        else:
            observation_tensor = {"observation": tf.constant(self.last_observation, tf.float32),
                                  "mask": tf.constant(self.allowed_actions, tf.bool),
                                  "integer": self.integers}
        self.goal_state_mask = tf.zeros((self.num_envs,), dtype=tf.bool)
        self.anti_goal_state_mask = tf.zeros((self.num_envs,), dtype=tf.bool)
        self.truncated = np.array(len(self.last_observation) * [False])
        self._current_time_step = ts.TimeStep(
            observation=observation_tensor,
            reward=self.reward_multiplier * self.reward,
            discount=self.discount,
            step_type=tf.convert_to_tensor([ts.StepType.MID] * self.num_envs, dtype=tf.int32))
        self.prev_dones = np.array(len(self.last_observation) * [False])

        return self._current_time_step

    def evaluate_simulator(self) -> ts.TimeStep:
        """Evaluates the simulator and returns the current time step. Primarily used to determine, whether the state is the last one or not."""
        self.flag_goal = tf.zeros((self.num_envs,), dtype=tf.bool)
        labels_mask = tf.convert_to_tensor(self.labels_mask, dtype=tf.bool)
        labels_mask = tf.reshape(labels_mask, (self.num_envs,))
        self.default_rewards = tf.constant(self.reward, dtype=tf.float32)
        antigoal_values_vector = self.antigoal_values_vector + self.default_rewards
        goal_values_vector = self.goal_values_vector + self.default_rewards
        self.goal_state_mask = labels_mask & self.dones
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
        self.reward += self.reward_shaping_rewards

        if self.flag_penalty:
            illegal_action_penalties = tf.where(
                self._played_illegal_actions,
                self.illegal_action_penalty,
                tf.zeros((self.num_envs,), dtype=tf.float32)
            )
            self.reward += illegal_action_penalties
        self.step_types = tf.where(
            still_running_mask,
            tf.where(
                self.prev_dones,
                self.init_step_types,
                self.default_step_types
            ),
            self.terminated_step_types
        )
        self._current_time_step = ts.TimeStep(
            step_type=self.step_types,
            reward=self.reward,
            discount=self.discount,
            observation=self.get_observation()
        )

        self.prev_dones = self.dones
        self.virtual_reward = self.reward - self.default_rewards
        return self._current_time_step

    def _do_step_in_simulator(self, actions) -> StepInfo:
        """Does the step in the Stormpy simulator.
            returns:
                tuple of new TimeStep and penalty for performed action.
        """
        if self.model_memory_size > 0:
            actions, self.memory_update = actions["simulator_action"], tf.convert_to_tensor(
                actions["memory_update"], dtype=tf.float32)
        self.prev_states = np.reshape(
            self.vectorized_simulator.simulator_states.vertices, (self.num_envs, 1))

        self.reward_shaping_rewards = self.reward_shaper_function(
            self.prev_states, actions)
        observations, rewards, done, truncated, allowed_actions, metalabels = self.vectorized_simulator.step(
            actions=actions)
        self.last_observation = observations
        self.states = self.vectorized_simulator.simulator_states
        self.allowed_actions = allowed_actions
        self.labels_mask = metalabels
        if True in self.labels_mask:
            self.goal = True
        else:
            self.goal = False
        self.reward = tf.constant(
            rewards.tolist(), dtype=tf.float32) * self.reward_multiplier
        self.dones = done
        self.truncated = truncated
        self.integers = tf.reshape(
            tf.gather(self.state_to_observation_map, self.states.vertices),
            (self.num_envs, 1)
        )

    def get_mask_of_played_illegal_actions(self, actions) -> tf.Tensor:
        """Returns the mask of played illegal actions. Used for evaluation of the environment."""
        rows = tf.range(tf.shape(self.allowed_actions)[0])
        gather_indices = tf.stack([rows, actions], axis=-1)
        is_action_allowed = tf.gather_nd(self.allowed_actions, gather_indices)
        return is_action_allowed

    def change_illegal_actions(self, actions, mask):
        """Changes the illegal actions to the nearest legal action with lower index with module after underflow given mask with allowed actions."""
        lowest_allowed_actions = tf.argmax(mask, axis=-1, output_type=tf.int32)
        rows = tf.range(tf.shape(mask)[0])
        gather_indices = tf.stack([rows, actions], axis=-1)
        is_action_allowed = tf.gather_nd(mask, gather_indices)

        new_actions = tf.where(
            is_action_allowed,
            actions,
            lowest_allowed_actions
        )
        return new_actions.numpy()

    def _step(self, action) -> ts.TimeStep:
        """Does the step in the environment. Important for TF-Agents and the TFPyEnvironment."""
        self.cumulative_num_steps += self.num_envs
        self.last_action = action
        self._do_step_in_simulator(action)
        if self.flag_penalty:
            self._played_illegal_actions = self.get_mask_of_played_illegal_actions(
                action)
        evaluated_step = self.evaluate_simulator()
        return evaluated_step

    def current_time_step(self) -> ts.TimeStep:
        return self._current_time_step

    def current_state(self):
        return self.vectorized_simulator.simulator_states

    def observation_spec(self) -> ts.tensor_spec:
        return self._observation_spec

    def action_spec(self) -> ts.tensor_spec:
        return self._action_spec

    def get_observation(self) -> dict[str: tf.Tensor]:
        encoded_observation = self.last_observation
        mask = self.allowed_actions
        if self.model_memory_size > 0:
            memory_update = tf.reshape(
                self.memory_update, shape=(self.num_envs, 1))
            encoded_observation = tf.concat(
                [encoded_observation, memory_update], axis=1)
            return {"observation": tf.constant(encoded_observation, dtype=tf.float32), "mask": tf.constant(mask, dtype=tf.bool),
                    "integer": self.integers}
        return {"observation": tf.constant(encoded_observation, dtype=tf.float32), "mask": tf.constant(mask, dtype=tf.bool),
                "integer": self.integers}

    def get_simulator_observation(self) -> int:
        observation = self.last_observation
        return observation
