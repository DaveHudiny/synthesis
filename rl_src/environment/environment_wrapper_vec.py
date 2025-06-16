# Description: This file contains the environment wrapper class that is used to interact with the Storm model and the RL agent.
# Author: David Hudák
# Login: xhudak03
# File: environment_wrapper.py

from environment.renderers.grid_like_renderer import GridLikeRenderer
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
from tools.specification_check import SpecificationChecker

from vec_storm.storm_vec_env import StormVecEnv

from tools.state_estimators import LSTMStateEstimator

import json

import tf_agents.policies.tf_policy as TFPolicy

from reward_machines.predicate_automata import PredicateAutomata, create_dummy_predicate_automata
from environment.go_explore_manager import GoExploreManager

import time


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
    if labels is None or len(labels) == 0:
        return -1.0
    last_reward = labels[-1]
    return rewards[last_reward]


class EnvironmentWrapperVec(py_environment.PyEnvironment):
    """The most important class in this project. It wraps the Stormpy simulator and provides the interface for the RL agent.
    """

    def __init__(self, stormpy_model: storage.SparsePomdp, args: ArgsEmulator, num_envs: int = 1, enforce_compilation: bool = False):
        """Initializes the environment wrapper.

        Args:
            stormpy_model: The Storm model to be used.
            args: The arguments from the command line or ArgsSimulator.
            q_values_table: The Q-values table for the Q-learning agent.
            num_envs: The number of environments for vectorization.
            state_based_oracle: The state-based oracle for the environment, expecting some memoryless MDP controller.
        """
        # print(stormpy_model)
        self.args = args
        super(EnvironmentWrapperVec, self).__init__()
        # self.batched = True
        # self.batch_size = num_envs
        self.num_envs = num_envs
        self.use_stacked_observations = args.use_stacked_observations
        self.stormpy_model = stormpy_model

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
            max_steps=args.max_steps, metalabels=metalabels, model_path=args.prism_model, enforce_recompilation=enforce_compilation)

        try:
            self.state_to_observation_map = tf.constant(
                stormpy_model.observations)
            self.observation_valuations = np.array(
                self.vectorized_simulator.simulator.observation_by_ids)
            _, first_indices = np.unique(
                self.state_to_observation_map, return_index=True)
        except:
            logger.error(
                "State to observation map not possible to initialize. Using the default one.")
            exit(1)
            self.state_to_observation_map = tf.constant(
                self.vectorized_simulator.simulator.state_observation_ids)
            self.observation_valuations = np.array(
                self.vectorized_simulator.simulator.state_values)
            self.first_indices = self.state_to_observation_map
        self.observations_to_states_map = np.zeros((stormpy_model.nr_observations),
                                                   dtype=np.int32)
        for observation in range(stormpy_model.nr_observations):
            self.observations_to_states_map[observation] = np.where(
                self.state_to_observation_map == observation)[0][0]
        try:
            self.grid_like_renderer = GridLikeRenderer(
                self.vectorized_simulator, self.get_model_name())
        except:
            logger.error("Grid-like renderer not possible to initialize.")
            self.grid_like_renderer = None
        self.vectorized_simulator.simulator.set_max_steps(args.max_steps)

        # If the vectorized simulator was pre-initialized, we need to set the number of environments.
        self.vectorized_simulator.set_num_envs(num_envs)
        self.vectorized_simulator.reset()
        logger.info("Vectorized simulator initialized with {} environments.".format(num_envs))
        # Default labels mask for the environment given that the number of metalabels is 1 ("goals").
        self.labels_mask = list([False] * self.num_envs)
        self.encoding_method = args.encoding_method

        # Initialization of the penalty for illegal actions.
        self.flag_penalty = args.flag_illegal_action_penalty
        self.illegal_action_penalty = tf.constant(
            [self.args.illegal_action_penalty_per_step] * self.num_envs, dtype=tf.float32)

        # Initialization of the rewards before simulation.
        self.reward = tf.constant(0.0, dtype=tf.float32)

        # Initialization of reward model.
        self.model_name = self.get_model_name()
        self.set_reward_model(self.model_name)

        self._current_time_step = None

        # Initialization of the TF Agents specifications
        self.set_action_labeling()

        # Initialization of the observation spec.
        self.create_specifications()
        if args.use_stacked_observations:
            self.stacked_observations = tf.zeros(
                (self.num_envs, self.observation_spec_len * self.observation_length_multiplier), )

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

        self.initialize_step_types()

        # Add reward shaping
        self.reward_shaper_function = lambda observation, _: tf.zeros(
            (observation.shape[0],), dtype=tf.float32)

        self.predicate_automata_obs = args.predicate_automata_obs
        self.predicate_automata_states = tf.zeros(
            (num_envs, 1), dtype=tf.float32)
        self.do_goal_explore = False
        self.go_explore_manager = None
        if self.args.predicate_automata_obs or self.args.go_explore or self.args.curiosity_automata_reward:
            self.predicate_automata = create_dummy_predicate_automata(
                self.vectorized_simulator.get_observation_labels())
        else:
            self.predicate_automata = None

        self.current_num_steps = tf.constant(
            [0] * self.num_envs, dtype=tf.float32)

    def initialize_step_types(self):
        """Initializes the step types for the environment."""
        self.init_step_types = tf.constant(
            [ts.StepType.FIRST] * self.num_envs, dtype=tf.int32)
        self.default_step_types = tf.constant(
            [ts.StepType.MID] * self.num_envs, dtype=tf.int32)
        self.terminated_step_types = tf.constant(
            [ts.StepType.LAST] * self.num_envs, dtype=tf.int32)

    def set_go_explore(self, predicate_automata: PredicateAutomata = None):
        if self.go_explore_manager is None:
            assert predicate_automata is not None, "Predicate automata must be provided for Go-Explore manager initialization."
            self.go_explore_manager = GoExploreManager(
                automata=predicate_automata,
                original_initial_state=self.vectorized_simulator.simulator.initial_state,
                buffer_size=3000)
        self.do_goal_explore = True

    def unset_go_explore(self):
        self.do_goal_explore = False

    def go_explore_add_state(self, automata_state):
        mdp_states = self.vectorized_simulator.simulator_states.vertices
        self.go_explore_manager.add_state_vec_mine(
            automata_states=automata_state.flatten(),
            mdp_states=mdp_states,
        )

    def set_predicate_automata_state(self, automata_state):
        self.predicate_automata_states = tf.reshape(
            automata_state, (self.num_envs, 1), name="predicate_automata_states")
        self.predicate_automata_states = tf.cast(
            self.predicate_automata_states, dtype=tf.float32)

    def predicate_automata_update(self, observation):
        old_states = self.predicate_automata_states
        new_states = self.predicate_automata.step(old_states, observation)
        if self.go_explore_manager is not None:
            self.go_explore_add_state(new_states)
        self.predicate_automata_states = new_states

    def set_basic_rewards(self):
        rew_list = list(self.stormpy_model.reward_models.keys())
        self.reward_multiplier = -1.0 if (len(rew_list) == 0  or not "rew" in rew_list[-1]) else 10.0
        self.antigoal_values_vector = tf.constant(
            [self.args.evaluation_antigoal] * self.num_envs, dtype=tf.float32)
        self.goal_values_vector = tf.constant(
            [self.args.evaluation_goal] * self.num_envs, dtype=tf.float32)

    def set_reachability_rewards(self):
        self.reward_multiplier = 0.0
        self.antigoal_values_vector = tf.constant(
            [0.0] * self.num_envs, dtype=tf.float32)
        self.goal_values_vector = tf.constant(
            [self.args.evaluation_goal] * self.num_envs, dtype=tf.float32)

    def set_maxizing_rewards(self):
        self.reward_multiplier = 10.0
        self.antigoal_values_vector = tf.constant(
            [0.0] * self.num_envs, dtype=tf.float32)
        self.goal_values_vector = tf.constant(
            [2.0] * self.num_envs, dtype=tf.float32)

    def set_minimizing_rewards(self):
        self.reward_multiplier = -10.0
        self.antigoal_values_vector = tf.constant(
            [0.0] * self.num_envs, dtype=tf.float32)
        self.goal_values_vector = tf.constant(  # Decreasing the goal value to make the optimization more reasonable
            [2.0] * self.num_envs, dtype=tf.float32)

    def set_reward_model(self, model_name):
        self.reward_models = {
            "network": self.set_minimizing_rewards,
            "drone": self.set_reachability_rewards,
            "refuel": self.set_reachability_rewards,
            "intercept": self.set_reachability_rewards,
            "evade": self.set_reachability_rewards,
            "rocks": self.set_minimizing_rewards,
            "geo": self.set_reachability_rewards,
            "mba": self.set_minimizing_rewards,
            "maze": self.set_maxizing_rewards,
            "avoid": self.set_reachability_rewards,
            "grid": self.set_reachability_rewards,

        }
        key_found = False
        for key in self.reward_models.keys():
            if key in model_name:
                self.reward_models[key]()
                key_found = True
                break
        if not key_found:
            self.set_basic_rewards()

        self.reward_signum = tf.sign(
            self.reward_multiplier) if self.reward_multiplier != 0.0 else tf.constant(-1.0)

        # Initialization of the discount factor for the environment.
        self.discount = tf.convert_to_tensor(
            [self.args.discount_factor] * self.num_envs, dtype=tf.float32)

    def set_state_based_oracle(self, state_based_oracle: TFPolicy, state_based_sim: StormVecEnv):
        self.state_based_oracle = state_based_oracle
        self.state_based_sim = state_based_sim
        # self.reward_multiplier = -1.0
        # self.antigoal_values_vector = tf.constant(
        #     [-400.0] * self.num_envs, dtype=tf.float32)
        self.oracle_reward = 50

    def unset_state_based_oracle(self):
        self.state_based_oracle = None
        # self.state_based_sim = None
        self.oracle_reward = 0.0

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
        self.observation_length_multiplier = self.args.trajectory_num_steps if self.use_stacked_observations else 1
        self.added_information_constant_size = int(
            self.env_see_reward + self.env_see_last_action + self.env_see_num_steps)
        if self.args.predicate_automata_obs:
            predicate_automata_obs_size = 1
        else:
            predicate_automata_obs_size = 0

        if self.encoding_method == "Valuations":
            try:
                observation_len = len(
                    self.vectorized_simulator.simulator.observations[0])
                self.observation_spec_len = (
                    observation_len + OBSERVATION_SIZE + predicate_automata_obs_size + self.added_information_constant_size)

                observation_spec = tensor_spec.TensorSpec(shape=(self.observation_length_multiplier * self.observation_spec_len,),
                                                          dtype=tf.float32, name="observation")
            except:
                logging.error(
                    "Valuation encoding not possible, currently not compatible. Probably model issue.")
                raise "Valuation encoding not possible, currently not compatible. Probably model issue."
        elif self.encoding_method == "MaskedValuations":
            try:
                action_mask_size = len(self.action_keywords)
                json_example = self.stormpy_model.observation_valuations.get_json(
                    0)
                parse_data = json.loads(str(json_example))
                observation_spec = tensor_spec.TensorSpec(
                    shape=(len(parse_data) + OBSERVATION_SIZE +
                           action_mask_size + predicate_automata_obs_size,),
                    dtype=tf.float32, name="observation")
            except:
                logging.error(
                    "Valuation encoding not possible, currently not compatible. Probably model issue.")
                raise "Valuation encoding not possible, currently not compatible. Probably model issue."
        else:
            raise ValueError("Encoding method currently not implemented")
        return observation_spec

    def create_specifications(self):
        """Creates the specifications for the environment. Important for TF-Agents."""
        self.env_see_reward = False  # self.args.env_see_reward
        self.env_see_last_action = False  # self.args.env_see_last_action
        self.env_see_num_steps = False  # self.args.env_see_num_steps

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

    def set_num_envs(self, num_envs: int):
        self.num_envs = num_envs
        self.vectorized_simulator.set_num_envs(num_envs)
        self.set_reward_model(self.model_name)
        self.initialize_step_types()
        self.current_num_steps = tf.constant(
            [0] * self.num_envs, dtype=tf.float32)
        self.reset()

    def get_observation_tensor(self, observation, mask, integers, memory=None):
        if self.encoding_method == "MaskedValuations":
            encoded_observation = tf.concat(
                [observation, tf.cast(mask, dtype=tf.float32)], axis=1)
        else:
            encoded_observation = observation
        return {"observation": encoded_observation, "mask": mask, "integer": integers}

    def _reset(self) -> ts.TimeStep:
        """Resets the environment. Important for TF-Agents, since we have to restart environment many times."""
        logger.info("Resetting the environment.")
        self.last_observation, self.allowed_actions, self.labels_mask = self._restart_simulator()
        if self.use_stacked_observations:
            self.stacked_observations = tf.zeros(
                (self.num_envs, self.observation_spec_len * (self.observation_length_multiplier - 1), ))
            self.stacked_observations = tf.concat(
                [self.last_observation, self.stacked_observations], axis=1)
        # self.integer_observations = self.vectorized_simulator.observations # TODO: implement it with proposed vectorized simulator
        self.last_action = np.zeros((self.num_envs,), dtype=np.float32)
        self.virtual_reward = tf.zeros((self.num_envs,), dtype=tf.float32)
        self.dones = np.array(len(self.last_observation) * [False])
        self.orig_reward = tf.constant(
            np.array(len(self.last_observation) * [0.0]), dtype=tf.float32)
        hidden_state = self.vectorized_simulator.simulator_states
        self.integers = tf.reshape(
            tf.gather(self.state_to_observation_map, hidden_state.vertices),
            (self.num_envs, 1)
        )
        if hasattr(self, "state_estimator") and self.state_estimator is not None:
            self.state_estimator.reset()
        self.current_num_steps = tf.constant(
            [0] * self.num_envs, dtype=tf.float32)
        observation_tensor = self.get_observation()
        self.goal_state_mask = tf.zeros((self.num_envs,), dtype=tf.bool)
        self.anti_goal_state_mask = tf.zeros((self.num_envs,), dtype=tf.bool)
        self.truncated = np.array(len(self.last_observation) * [False])
        self._current_time_step = ts.TimeStep(
            observation=observation_tensor,
            reward=self.reward_multiplier * self.orig_reward,
            discount=self.discount,
            step_type=tf.convert_to_tensor([ts.StepType.FIRST] * self.num_envs, dtype=tf.int32))
        self.prev_dones = np.array(len(self.last_observation) * [False])

        return self._current_time_step

    def get_oracle_based_reward(self, action, sim_state):
        # Compute observations
        vertices = tf.reshape(sim_state.vertices, (-1, 1))
        vertices = tf.cast(vertices, dtype=tf.int32)

        observations = tf.gather_nd(
            self.state_based_sim.simulator.observations, vertices)
        # Predict actions by the oracle
        fake_time_step = ts.TimeStep(
            step_type=tf.convert_to_tensor(
                [ts.StepType.MID] * self.num_envs, dtype=tf.int32),
            reward=tf.zeros((self.num_envs,), dtype=tf.float32),
            discount=tf.ones((self.num_envs,), dtype=tf.float32),
            observation=observations
        )
        oracle_actions = self.state_based_oracle(fake_time_step).action
        oracle_actions = tf.cast(oracle_actions, dtype=tf.int32)
        oracle_actions = self.change_illegal_actions_to_random_allowed(
            oracle_actions, self.allowed_actions)
        # Compute the reward
        oracle_reward = tf.where(
            tf.equal(oracle_actions, action),
            tf.constant(self.oracle_reward, dtype=tf.float32),
            tf.constant(-self.oracle_reward, dtype=tf.float32)
        )
        # self.oracle_reward *= self.args.discount_factor
        return oracle_reward

    def evaluate_simulator(self) -> ts.TimeStep:
        """Evaluates the simulator and returns the current time step. Primarily used to determine, whether the state is the last one or not."""
        self.flag_goal = tf.zeros((self.num_envs,), dtype=tf.bool)
        labels_mask = tf.convert_to_tensor(self.labels_mask, dtype=tf.bool)
        labels_mask = tf.reshape(labels_mask, (self.num_envs,))
        self.default_rewards = tf.constant(
            self.orig_reward, dtype=tf.float32) * self.reward_multiplier
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
        # self.reward = tf.where(
        #     self.goal_state_mask,
        #     goal_values_vector,
        #     tf.where(
        #         still_running_mask,
        #         self.default_rewards,
        #         antigoal_values_vector)
        # )
        self.reward += tf.abs(self.reward_multiplier) * \
            self.reward_shaping_rewards
        if hasattr(self, "state_based_oracle") and hasattr(self, "state_based_sim") and self.state_based_oracle is not None:

            self.reward += self.get_oracle_based_reward(
                self.last_action, self.current_state())
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
        self.stacked_observations = tf.concat(
            [self.last_observation, self.stacked_observations[:, :-self.observation_spec_len]], axis=1) if self.use_stacked_observations else self.last_observation
        # Nullify all stacked observations for environments that are done
        if self.use_stacked_observations:
            self.stacked_observations = tf.where(
                tf.reshape(self.prev_dones, (-1, 1)), tf.zeros_like(self.stacked_observations), self.stacked_observations)
            # print(self.stacked_observations)
        self.current_num_steps = tf.where(
            self.prev_dones,
            tf.zeros_like(self.current_num_steps),
            self.current_num_steps + 1
        )
        self._current_time_step = ts.TimeStep(
            step_type=self.step_types,
            reward=self.reward,
            discount=self.discount,
            observation=self.get_observation()
        )

        self.prev_dones = self.dones
        self.virtual_reward = self.reward
        # To better demonstrate the objective reward
        self.orig_reward = self.orig_reward * self.reward_signum
        return self._current_time_step

    def temporarily_set_num_envs(self, num_envs: int):
        self.orig_num_envs = self.num_envs
        self.num_envs = num_envs
        self.vectorized_simulator.set_num_envs(num_envs)
        self.set_reward_model(self.model_name)
        self.initialize_step_types()
        self.reset()

    def reset_num_envs(self):
        """Resets the number of environments to the original value."""
        if hasattr(self, "orig_num_envs"):
            self.set_num_envs(self.orig_num_envs)
        else:
            logger.warning(
                "No original number of environments set. Cannot reset.")

    def _go_explore_resets(self, prev_dones):
        """Changes the state of the simulator to the state defined by the Go-Explore manager, when the episode just started."""
        new_states = self.go_explore_manager.sample_states(self.num_envs)
        original_states = self.vectorized_simulator.simulator_states.vertices
        new_states = np.where(prev_dones, new_states, original_states)

        self.vectorized_simulator.set_states(new_states)

        return self.vectorized_simulator.no_step()

    def _do_step_in_simulator(self, actions) -> StepInfo:
        """Does the step in the Stormpy simulator.
            returns:
                tuple of new TimeStep and penalty for performed action.
        """
        self.prev_states = np.reshape(
            self.vectorized_simulator.simulator_states.vertices, (self.num_envs, 1))
        self.reward_shaping_rewards = self.reward_shaper_function(
            self.prev_states, actions)
        observations, rewards, done, truncated, allowed_actions, metalabels = self.vectorized_simulator.step(
            actions=actions)
        if self.predicate_automata is not None:
            self.predicate_automata_update(observations)
            if self.do_goal_explore:
                observations, rewards, done, truncated, allowed_actions, metalabels = self._go_explore_resets(
                    prev_dones=self.prev_dones)
        # if there any truncated episodes, we should set the done flag to True
        self.last_observation = observations
        self.states = self.vectorized_simulator.simulator_states
        self.allowed_actions = allowed_actions
        self.labels_mask = metalabels
        self.orig_reward = tf.constant(
            rewards.tolist(), dtype=tf.float32)
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

    @staticmethod
    def change_illegal_actions_to_random_allowed(actions, masks, dones=None):
        """Changes the illegal actions to random allowed actions given mask with allowed actions."""
        rows = tf.range(tf.shape(masks)[0])
        gather_indices = tf.stack([rows, actions], axis=-1)
        is_action_allowed = tf.gather_nd(masks, gather_indices)
        batch_size = tf.shape(masks)[0]
        flat_indices = tf.where(masks)
        legal_counts = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)
        batch_offsets = tf.cumsum(tf.concat([[0], legal_counts[:-1]], axis=0))
        random_offsets = tf.random.uniform(
            shape=(batch_size,),
            maxval=tf.reduce_max(legal_counts),
            dtype=tf.int32
        ) % legal_counts
        selected_flat_indices = batch_offsets + random_offsets
        selected_actions = tf.gather(flat_indices[:, 1], selected_flat_indices)
        selected_actions = tf.cast(selected_actions, tf.int32)
        new_actions = tf.where(
            is_action_allowed,
            actions,
            selected_actions
        )
        return new_actions.numpy(), tf.logical_not(is_action_allowed)

    def _step(self, action) -> ts.TimeStep:
        """Does the step in the environment. Important for TF-Agents and the TFPyEnvironment."""
        # current_time = time.time()

        action, illegals = self.change_illegal_actions_to_random_allowed(
            action, self.allowed_actions, self.dones)
        self._played_illegal_actions = illegals
        self.cumulative_num_steps += self.num_envs
        self.last_action = tf.cast(action, dtype=tf.float32)
        self._do_step_in_simulator(action)
        evaluated_step = self.evaluate_simulator()
        return evaluated_step

    def get_model_name(self):
        # args.prism_model contains .../model_name/sketch.templ" so we need to extract the model name
        if self.args.model_name != "":
            model_name = self.args.model_name
        elif self.args.prism_model is not None:
            model_name = self.args.prism_model.split("/")[-2]
        else:
            model_name = "unknown"
        return model_name

    # Search from list of known models and return True if the model is renderable
    def is_renderable(self):
        renderable_models = ["mba", "mba-small", "drone-2-6-1", "drone-2-8-1", "geo-2-8",
                             "refuel-10", "refuel-20", "intercept", "super-intercept", "evade",
                             "rocks-16", "rocks-4-20"]
        return self.get_model_name() in renderable_models

    def render(self, mode='rgb_array', trajectory=None):
        """Returns the renderable image of the environment. Supposed to be added to a batch of renders"""
        if self.is_renderable():
            return self.grid_like_renderer.render(mode, trajectory)
        else:
            return None

    def current_time_step(self) -> ts.TimeStep:
        return self._current_time_step

    def current_state(self):
        return self.vectorized_simulator.simulator_states

    def observation_spec(self) -> ts.tensor_spec:
        return self._observation_spec

    def action_spec(self) -> ts.tensor_spec:
        return self._action_spec

    def get_observation(self) -> dict[str: tf.Tensor]:
        encoded_observation = self.last_observation if not self.use_stacked_observations else self.stacked_observations
        mask = self.allowed_actions
        integers = self.integers
        if self.encoding_method == "MaskedValuations":
            encoded_observation = tf.concat(
                [encoded_observation, tf.cast(mask, dtype=tf.float32)], axis=1)
        if self.env_see_reward:
            encoded_observation = tf.concat(
                [encoded_observation, tf.reshape(self.orig_reward, (-1, 1))], axis=1)
        if self.env_see_last_action:
            encoded_observation = tf.concat(
                [encoded_observation, tf.reshape(self.last_action, (-1, 1))], axis=1)
        if self.env_see_num_steps:
            encoded_observation = tf.concat(
                [encoded_observation, tf.reshape(self.current_num_steps, (-1, 1))], axis=1)
        if self.predicate_automata_obs:
            predicate_automata_obs = self.predicate_automata_states
            encoded_observation = tf.concat(
                [encoded_observation, predicate_automata_obs], axis=1)
        encoded_observation = tf.cast(encoded_observation, dtype=tf.float32)
        return {"observation": encoded_observation, "mask": tf.constant(mask, dtype=tf.bool),
                "integer": integers}

    def encode_observation(self, integer_observation, memory, state) -> dict[str: tf.Tensor]:
        """Encodes the observation based on the integer observation and memory."""
        # json_valuation = json.loads(str(self.observation_valuations.get_json(integer_observation)))
        # valuated = np.array(list(json_valuation.values()), dtype=np.float32)
        valuated = self.observation_valuations[integer_observation]
        valuated = tf.constant(valuated, dtype=tf.float32)
        allowed_actions = self.vectorized_simulator.simulator.allowed_actions[state]
        return {"observation": tf.constant(valuated, dtype=tf.float32),
                "mask": tf.constant(allowed_actions, tf.bool),
                "integer": tf.constant([integer_observation], dtype=tf.int32)}

    def create_fake_timestep_from_valuations(self, valuations):
        """Creates a fake TimeStep from the valuations."""
        observation = tf.constant([valuations], dtype=tf.float32)
        mask = tf.constant([True] * self.nr_actions, dtype=tf.bool)
        integer = tf.constant([0], dtype=tf.int32)
        time_step = ts.TimeStep(
            observation={"observation": [observation],
                         "mask": [mask], "integer": [integer]},
            reward=tf.constant([0.0], dtype=tf.float32),
            discount=tf.constant([1.0], dtype=tf.float32),
            step_type=tf.constant([ts.StepType.MID], dtype=tf.int32)
        )
        return time_step

    def create_fake_timestep_from_observation_integer(self, observation_integer: int | list[int]) -> ts.TimeStep:
        """Creates a fake TimeStep from the observation integer."""
        # observation = create_valuations_encoding(observation_integer, self.stormpy_model)
        if isinstance(observation_integer, int):
            observation_integer = [observation_integer]
        observation = self.observation_valuations[observation_integer]
        if self.observation_length_multiplier > 1:
            observation = tf.tile(
                observation, [1, self.observation_length_multiplier])
        observation = tf.constant(observation, dtype=tf.float32)
        # state = np.where(self.state_to_observation_map ==
        #                      observation_integer)
        state = self.observations_to_states_map[observation_integer]
        # if self.vectorized_simulator.simulator.sinks[state]:
        #     mask = tf.ones(shape=(len(observation_integer), self.nr_actions,), dtype=tf.bool)
        # mask = tf.constant([[True] * self.nr_actions], dtype=tf.bool)
        # else:
        #     mask = tf.constant(self.vectorized_simulator.simulator.allowed_actions[state].tolist(), dtype=tf.bool)
        mask = tf.constant(
            self.vectorized_simulator.simulator.allowed_actions[state].tolist(), dtype=tf.bool)
        sinks = self.vectorized_simulator.simulator.sinks[state]
        sinks = tf.reshape(sinks, (-1, 1))
        mask = tf.where(sinks, tf.ones(
            shape=(len(observation_integer), self.nr_actions,), dtype=tf.bool), mask)
        integer = tf.constant(observation_integer, dtype=tf.int32)
        integer = tf.reshape(integer, (-1, 1))
        mask = tf.reshape(mask, (-1, self.nr_actions))
        # observation = tf.reshape(observation, len(observation_integer), -1)

        time_step = ts.TimeStep(
            observation={"observation": observation,
                         "mask": mask, "integer": integer},
            reward=tf.zeros(integer.shape[0], dtype=tf.float32),
            discount=tf.ones(integer.shape[0], dtype=tf.float32),
            step_type=tf.fill(integer.shape[0], ts.StepType.MID)
        )
        return time_step

    def get_simulator_observation(self) -> int:
        observation = self.last_observation
        return observation
    
    def get_state(self, use_features : bool = True) -> tf.Tensor:
        """Returns the current state of the environment."""
        if use_features:
            states_vertices = self.vectorized_simulator.simulator_states.vertices
            states_values = tf.constant(self.vectorized_simulator.simulator.state_values[states_vertices].tolist(), tf.float32)
            return states_values
        else:
            return tf.constant(self.vectorized_simulator.simulator_states.vertices.tolist())


def test_environment():
    pass


if __name__ == "__main__":
    exit(test_environment())
