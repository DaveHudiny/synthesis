from vec_storm.storm_vec_env import StormVecEnv

from environment.vectorized_sim_initializer import SimulatorInitializer

import tensorflow as tf
import numpy as np

from keras import Model as KerasModel

from keras.layers import LSTM, TimeDistributed, Dense

from tf_agents.policies import TFPolicy
from tf_agents.trajectories.time_step import TimeStep

from tools.args_emulator import ArgsEmulator


def generate_legal_actions(masks):
    """Changes the illegal actions to random allowed actions given mask with allowed actions."""
    batch_size = tf.shape(masks)[0]
    flat_indices = tf.where(masks)
    legal_counts = tf.reduce_sum(tf.cast(masks, tf.int32), axis=1)
    batch_offsets = tf.cumsum(tf.concat([[0], legal_counts[:-1]], axis=0))
    legal_counts_zero = tf.cast(legal_counts == 0, tf.int32)
    legal_counts = legal_counts + legal_counts_zero
    random_offsets = tf.random.uniform(
        shape=(batch_size,),
        maxval=tf.reduce_max(legal_counts),
        dtype=tf.int32
    ) % legal_counts
    selected_flat_indices = batch_offsets + random_offsets
    selected_actions = tf.gather(flat_indices[:, 1], selected_flat_indices)
    selected_actions = tf.cast(selected_actions, tf.int32)
    return selected_actions.numpy()


def change_illegal_actions_to_random_allowed(actions, masks):
    """Changes the illegal actions to random allowed actions given mask with allowed actions."""
    rows = tf.range(tf.shape(masks)[0])
    gather_indices = tf.stack([rows, actions], axis=-1)
    is_action_allowed = tf.gather_nd(masks, gather_indices)
    selected_actions = generate_legal_actions(masks)
    new_actions = tf.where(
        is_action_allowed,
        actions,
        selected_actions
    )
    return new_actions.numpy(), tf.logical_not(is_action_allowed)


class StateEstimator:
    def __init__(self, env: StormVecEnv):
        self.env = env


class LSTMNetwork(KerasModel):
    def __init__(self, env: StormVecEnv):
        super().__init__()
        self.num_of_extra_features = self.compute_obs_len_diff(env)
        self.lstm = LSTM(128, return_sequences=True, return_state=True)
        self.time_distributed = TimeDistributed(
            Dense(self.num_of_extra_features))

    def compute_obs_len_diff(self, env: StormVecEnv):
        union_of_labels = set(env.get_state_labels()).union(
            set(env.get_observation_labels()))
        union_len = len(union_of_labels)
        num_of_extra_features = union_len - len(env.get_observation_labels())
        if num_of_extra_features <= 0:
            raise ValueError(
                "The state-based environment has less or same number of features as the environment.")
        return num_of_extra_features

    def call(self, inputs, prev_hidden_state=None):
        x, _, _ = self.lstm(inputs, prev_hidden_state)
        x = self.time_distributed(x)
        return x

    def predict_single_element_from_sequence(self, input, prev_hidden_state):
        x, hidden_state, cell_state = self.lstm(input, prev_hidden_state)
        x = self.time_distributed(x)
        return x, (hidden_state, cell_state)


class LSTMStateEstimator(StateEstimator):

    def __init__(self, env: StormVecEnv):
        super().__init__(env)
        self.model = LSTMNetwork(env)
        self.model.compile(optimizer='adam', loss='mse')
        self.__init_indices_of_extra_features()
        self.__init_observations_state_based()
        self.history = None

    def reset(self):
        """
        Resets the history of the model.
        """
        self.history = None

    def collect_and_train(self, num_steps: int, external_policy: TFPolicy = None, epochs=50):
        xs, ys = self.collect_data(
            num_steps=num_steps, external_policy=external_policy)
        self.model.fit(xs, ys, epochs=epochs)
        # predicted_val, hidden_state = self.model.predict_single_element_from_sequence(tf.reshape(xs[0, 0:10, :], (1, 10, -1)), None)
        # print(predicted_val)
        # predicted_val, hidden_state = self.model.predict_single_element_from_sequence(tf.reshape(xs[0, 10, :], (1, 1, -1)), hidden_state)
        # print(predicted_val)
        # self.model.summary()

    def step_wrapper(self, action, prev_action, batch_size):
        time_step = self.env.step(action)
        observation = tf.concat([time_step[0], prev_action], axis=1)
        observation = tf.reshape(observation, (batch_size, 1, -1))
        missing_features = self.get_missing_features()
        missing_features = tf.reshape(missing_features, (batch_size, 1, -1))
        allowed_actions = time_step[4]
        self.prev_action = action
        return observation, missing_features, allowed_actions

    def reset_wrapper(self, batch_size):
        time_step = self.env.reset()
        allowed_actions = time_step[1]
        prev_action = np.zeros((batch_size, 1,))
        observation = tf.concat([time_step[0], prev_action], axis=1)
        observation = tf.reshape(observation, (batch_size, 1, -1))
        observation = tf.cast(observation, tf.float32)
        missing_features = self.get_missing_features()
        missing_features = tf.cast(missing_features, tf.float32)
        missing_features = tf.reshape(missing_features, (batch_size, 1, -1))
        return observation, missing_features, allowed_actions

    def play_tf_policy(self, observation: tf.Tensor, missing_features: tf.Tensor, batch_size: int = 256, external_policy: TFPolicy = None, policy_state = None):
        if len(observation.shape) == 3:
            observation = tf.reshape(observation, (batch_size, -1))
        observation = observation[:, :-1] # Remove the last action
        if len(missing_features.shape) == 3:
            missing_features = tf.reshape(missing_features, (batch_size, -1))
        fake_time_step = TimeStep(
            step_type=tf.constant([1] * batch_size, dtype=tf.int32),
            reward=tf.constant([0] * batch_size, dtype=tf.float32),
            discount=tf.constant([1] * batch_size, dtype=tf.float32),
            observation=tf.concat([observation, missing_features], axis=1))
        policy_step = external_policy.action(fake_time_step, policy_state)
        policy_state = policy_step.state
        action = policy_step.action
        return action, policy_state

    def collect_data(self, external_policy: TFPolicy = None, batch_size=256, num_steps=1000):
        observations = []
        missing_features_arr = []
        prev_action = np.zeros((batch_size, 1,))
        observation, missing_features, allowed_actions = self.reset_wrapper(
            batch_size)
        observations.append(observation)
        missing_features_arr.append(missing_features)
        if external_policy is not None:
            policy_state = external_policy.get_initial_state(batch_size)
        for i in range(num_steps):
            if external_policy is not None:
                # external_policy.action(observation)
                action, policy_state = self.play_tf_policy(
                    observation, missing_features, batch_size, external_policy, policy_state=policy_state)
                action, _ = change_illegal_actions_to_random_allowed(
                    action, allowed_actions)
            else:
                action = generate_legal_actions(allowed_actions)
            observation, missing_features, allowed_actions = self.step_wrapper(
                action, prev_action, batch_size)
            observations.append(observation)
            missing_features_arr.append(missing_features)
            prev_action = np.reshape(action, (batch_size, 1))
        concatenated_observations = tf.concat(observations, axis=1)
        concatenated_missing_features = tf.concat(missing_features_arr, axis=1)
        return concatenated_observations, concatenated_missing_features

    def __init_indices_of_extra_features(self):
        labels_original = self.env.get_observation_labels()
        labels_state_based = self.env.get_state_labels()
        indices = []
        for i, label in enumerate(labels_state_based):
            if label not in labels_original:
                indices.append(i)
        self.indices = indices

    def __init_observations_state_based(self):
        self.state_values = self.env.simulator.state_values

    def get_missing_features(self):
        states = self.env.simulator_states
        # Gather observations for states from state_based_env
        state_based_env_obs = self.state_values[states.vertices]
        # Gather missing features
        missing_features = state_based_env_obs[:, self.indices]
        return missing_features

    def estimate_missing_features(self, observations : tf.Tensor, action):
        observations = tf.reshape(observations, (observations.shape[0], 1, -1))
        observations = tf.concat([observations, tf.reshape(action, (observations.shape[0], 1, -1))], axis=2)
        missing_features, history = self.model.predict_single_element_from_sequence(observations, self.history)
        missing_features = tf.reshape(missing_features, (missing_features.shape[0], -1))
        self.history = history
        return missing_features


if __name__ == "__main__":
    env = SimulatorInitializer.try_load_simulator_by_name_from_pickle(
        "refuel-20", "compiled_models_vec_storm")
    state_estimator = LSTMStateEstimator(env)
