import numpy as np
import tensorflow as tf

from stormpy import simulator
from stormpy import storage
import stormpy

from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step_spec
from agents.tools import *
from environment.reward_shaping_models import *

import json
import argparse

OBSERVATION_SIZE = 0  # Constant for valuation encoding
MAXIMUM_SIZE = 6  # Constant for reward shaping

import logging
logging.basicConfig(level=logging.INFO)


class Environment_Wrapper(py_environment.PyEnvironment):
    def __init__(self, stormpy_model: storage.SparsePomdp, args: argparse.Namespace):
        super(Environment_Wrapper, self).__init__()
        self.stormpy_model = stormpy_model
        self.simulator = simulator.create_simulator(self.stormpy_model)
        self.labels = list(self.simulator._report_labels())
        self.args = args
        self.encoding_method = args.encoding_method
        self.goal_value = tf.constant(args.evaluation_goal, dtype=tf.float32)
        self.antigoal_value = tf.constant(args.evaluation_antigoal,
                                          dtype=tf.float32)
        self.discount = tf.constant(args.discount_factor, dtype=tf.float32)
        self.reward = tf.constant(0.0, dtype=tf.float32)
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
        self.using_logits = args.using_logits
        self.create_specifications()
        if self.using_logits:
            self.action_convertor = self._convert_action_with_logits
        else:
            self.action_convertor = self._convert_action

        self.last_action = 0
        self.visited_states = []
        self.empty_reward = False
        self.special_labels = ["(((sched = 0) & (t = (8 - 1))) & (k = (20 - 1)))", "goal", "done", "((x = 2) & (y = 0))"]
        self.virtual_value = tf.constant(0.0, dtype=tf.float32)

    def select_reward_shaping_function(self):
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

    def create_observation_spec(self):
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

        else:
            raise ValueError("Encoding method not recognized")
        return observation_spec[0]

    def create_specifications(self):
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
        if self.using_logits:
            self._action_spec = {"action": tensor_spec.BoundedTensorSpec(
                shape=(),
                dtype=tf.int32,
                minimum=0,
                maximum=len(self.action_keywords) - 1,
                name="action"
            ), "logits": tensor_spec.TensorSpec(
                shape=(len(self.action_keywords),),
                dtype=tf.float32,
                name="logits"
            )}
        else:
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

    def compute_mask(self):
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

    def create_encoding(self, observation):
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

    def _reset(self):
        self._finished = False
        self._num_steps = 0
        stepino = self.simulator.restart()
        self.labels = list(self.simulator._report_labels())
        self.virtual_value = tf.constant(0.0, dtype=tf.float32)
        observation = stepino[0]
        if stepino[1] == []:
            self.empty_reward = True
            reward = tf.constant(-1.0, dtype=tf.float32)
        else:
            reward = tf.constant(stepino[1][0], dtype=tf.float32)
        if not self.action_filtering:
            mask = self.compute_mask()
            observation_vector = self.create_encoding(observation)
            observation_tensor = {
                "observation": observation_vector, "mask": tf.constant(mask[0], dtype=tf.bool), "integer": tf.constant([observation], dtype=tf.int32)}
        else:
            observation_tensor = self.create_encoding(observation)
        self._current_time_step = ts.TimeStep(
            observation=observation_tensor, reward=-reward, discount=self.discount, step_type=ts.StepType.FIRST)
        
        return self._current_time_step

    def _convert_action(self, action):
        act_keyword = self.act_to_keywords[int(action)]
        choice_list = self.get_choice_labels()
        action = choice_list.index(act_keyword)
        return action

    def _convert_action_with_logits(self, action):
        logits = action["logits"]
        action = action["action"]

        if self.is_legal_action(action):
            self.last_action = action
            action = self._convert_action(action)
        else:
            actions = tf.argsort(logits, direction='DESCENDING')
            for act in actions:
                if self.is_legal_action(act):
                    action = act
                    break
            self.last_action = action
            action = self._convert_action(action)
        return action

    def compute_square_root_distance_from_goal(self):
        self.simulator.set_observation_mode(
            stormpy.simulator.SimulatorObservationMode.PROGRAM_LEVEL)
        json_final = json.loads(str(self.simulator._report_state()))
        ax = json_final["dx"]
        ay = json_final["dy"]
        distance = (MAXIMUM_SIZE - ax) + (MAXIMUM_SIZE - ay)
        self.simulator.set_observation_mode(
            stormpy.simulator.SimulatorObservationMode.STATE_LEVEL)
        return distance
    
    def get_coordinates(self):
        self.simulator.set_observation_mode(
            stormpy.simulator.SimulatorObservationMode.PROGRAM_LEVEL)
        json_final = json.loads(str(self.simulator._report_state()))
        ax = json_final["ax"]
        ay = json_final["ay"]
        self.simulator.set_observation_mode(
            stormpy.simulator.SimulatorObservationMode.STATE_LEVEL)
        return ax, ay
    
    def is_goal_state(self, labels):
        for label in labels:
            if label in self.special_labels:
                return True
        return False

    def evaluate_simulator(self):
        self.labels = list(self.simulator._report_labels())
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
            self.virtual_value = self.goal_value
            self._current_time_step = ts.termination(
                observation=self.get_observation(), reward=self.goal_value + self.reward)
        elif self.simulator.is_done() and "traps" in self.labels:
            # print("Trapped!")
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

    def is_legal_action(self, action):
        act_keyword = self.act_to_keywords[int(action)]
        choice_list = self.get_choice_labels()
        if act_keyword in choice_list:
            return True
        else:
            return False

    def get_random_legal_action(self):
        available_actions = self.simulator.available_actions()
        return np.random.choice(available_actions)

    def get_max_step_finish_timestep(self):
        if self.reward_shaping:
            distance = self.compute_square_root_distance_from_goal()
            return ts.termination(observation=self.get_observation(),
                                  reward=tf.constant((self.goal_value / distance) / MAXIMUM_SIZE, dtype=tf.float32))
        else:
            return ts.termination(observation=self.get_observation(), reward=tf.constant(self.reward, dtype=tf.float32))

    def _do_step(self, action):
        penalty = 0.0
        self._num_steps += 1
        self.last_action = action
        if not self.action_filtering:
            action = self.action_convertor(action)
            stepino = self.simulator.step(int(action))
        else:
            if self.using_logits:
                action = self.action_convertor(action)
            elif not self.is_legal_action(action):
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

    def _step(self, action):
        self.virtual_value = tf.constant(0.0, dtype=tf.float32)
        if self._finished:
            self._finished = False
            return self._reset()
        stepino, penalty = self._do_step(action)
        simulator_reward = stepino[1][-1] if not self.empty_reward else 1.0
        if "traps" in list(self.simulator._report_labels()):
            self.reward = self.antigoal_value
        else:
            self.reward = tf.constant(
                -simulator_reward + penalty, dtype=tf.float32)
        return self.evaluate_simulator()

    def current_time_step(self) -> ts.TimeStep:
        return self._current_time_step

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def get_observation(self):
        observation = self.simulator._report_observation()
        if not self.action_filtering:
            mask = self.compute_mask()
            return {"observation": self.create_encoding(observation), "mask": tf.constant(mask[0], dtype=tf.bool), 
                    "integer": tf.constant([observation], dtype=tf.int32)}
        else:
            return self.create_encoding(observation)

    def get_choice_labels(self):
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

    def get_simulator_observation(self):
        observation = self.simulator._report_observation()
        return observation


if __name__ == "__main__":
    pass
