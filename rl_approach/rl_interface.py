# File: rl_interface.py
# Author: David Hudák
# Purpose: Interface between paynt and safe reinforcement learning algorithm

import sys

import numpy as np
sys.path.append('/home/david/Plocha/paynt/synthesis/rl_approach/safe_rl')

import pickle
import dill

import shield_v2
import rl_simulator_v2 as rl_simulator
import tensorflow as tf
import tf_agents as tfa
import stormpy

import logging
logger = logging.getLogger(__name__)




class RLInterface:
    def __init__(self):
        logger.debug("Creating interface between paynt and RL algorithms.")
        self._model = None
        self.unique_observations = None

    def add_to_sys_argv_obstacle():
        sys.argv.append("-m")
        sys.argv.append("obstacle")
        sys.argv.append("-c")
        sys.argv.append("N=6")
    
    def add_to_sys_argv_refuel():
        sys.argv.append("-m")
        sys.argv.append("refuel")
        sys.argv.append("-c")
        sys.argv.append("N=6,ENERGY=10")
    
    def add_to_sys_argv_intercept():
        sys.argv.append("-m")
        sys.argv.append("intercept")
        sys.argv.append("-c")
        sys.argv.append("N=5,RADIUS=2")

    def add_to_sys_argv_evade():
        sys.argv.append("-m")
        sys.argv.append("evade")
        sys.argv.append("-c")
        sys.argv.append("N=6,RADIUS=2")

    def add_to_sys_argv_sac_strategy():
        sys.argv.append("--learning_method")
        sys.argv.append("SAC")

    def add_to_sys_argv_ddqn_strategy():
        sys.argv.append("--learning_method")
        sys.argv.append("DDQN")

    

    # def convert_state_storm_tensor(self, storm_state):
    #     discount = ts.tensor(storm_state.discount)
    #     tfa.trajectories.time_step.TimeStep(discount=discount, reward=reward, observation=observation, step_type=step_type)
    #     pass

    def convert_state_tensor_storm_action(self, tensor_state):
        pass

    def save_model(self, path_to_model):
        with open(path_to_model, "wb") as file:
            logger.debug(f"Saving model to {path_to_model}")
            dill.dump(self._model, file)
            logger.debug(f"Saved model to {path_to_model}")

    def load_model(self, path_to_model):
        with open(path_to_model, "rb") as file:
            logger.debug(f"Loading model from {path_to_model}")
            self._model = dill.load(file)
            logger.debug(f"Loaded model from {path_to_model}")

    def ask_model(self, storm_state):
        tensor_state = None
        self._model.agent.policy.action(tensor_state)

    def create_model(self):
        RLInterface.add_to_sys_argv_evade()
        RLInterface.add_to_sys_argv_ddqn_strategy()
        self._model, self.storm_model, self.tf_environment = shield_v2.improved_main()
        

    def train_model(self):
        pass

    def create_timestep(self, mask_size):
        discount = tf.constant([1.0], dtype=tf.float32)
        observation = {
            'mask': tf.constant([[False for i in range(mask_size)]], dtype=tf.bool),
            'obs': tf.constant([[0]], dtype=tf.int32)
        }
        reward = tf.constant([-1.0], dtype=tf.float32)
        step_type = tf.constant([1], dtype=tf.int32)

        time_step = tfa.trajectories.time_step.TimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=observation
        )
        return time_step
    
    def initialize_example_policy_steps(self):
        action_1 = tf.constant([5], dtype=tf.int32)
        state_1 = [tf.constant([[-2.91052902e-06, -8.85531008e-18,  5.82103610e-01,
         7.61594057e-01, -2.36220972e-15,  9.75167040e-06,
        -4.57815673e-07, -1.68067520e-04, -3.48037747e-06,
         4.70267087e-01,  7.61063814e-01, -6.88805163e-01,
        -6.32826388e-01,  5.12652214e-08, -7.61101961e-01,
         1.99017022e-03, -7.00955510e-01,  1.55778164e-02,
         7.43498802e-01, -5.07930417e-05, -7.51618445e-01,
         5.99638361e-06,  2.40339378e-16, -1.52101357e-22,
         4.01244142e-13,  5.36106826e-10,  1.27360111e-14,
        -3.07568138e-11, -2.29704846e-02, -5.96691564e-07,
         7.44417982e-12,  2.82101035e-01, -7.61594176e-01,
        -6.58637703e-07, -1.23091155e-11, -8.76696618e-08,
         4.09818022e-05, -2.28915014e-05, -3.99624067e-09,
         7.75590379e-05]], dtype=tf.float32), tf.constant([[-2.9106070e-06, -5.3992881e-09,  9.9999994e-01,  1.0000000e+00,
        -9.8564726e-01,  1.4256241e-02, -1.0000000e+00, -1.4914495e-03,
        -4.0949690e-06,  5.1041371e-01,  1.0000000e+00, -8.4835958e-01,
        -9.9998182e-01,  9.9437833e-01, -9.9967545e-01,  6.8625789e-03,
        -8.6917669e-01,  9.9999917e-01,  9.9991500e-01, -8.6903054e-01,
        -9.7666532e-01,  9.9999994e-01,  2.4134915e-16, -2.8098094e-20,
         2.8336157e-07,  9.9995887e-01,  1.3581178e-01, -3.0829502e-11,
        -1.0000000e+00, -9.1760963e-01,  7.4441798e-12,  2.9200581e-01,
        -1.0000000e+00, -8.0229338e-06, -2.0592861e-06, -1.0000000e+00,
         4.0981802e-05, -2.2891501e-05, -8.0547339e-01,  7.7559038e-05]], dtype=tf.float32)]
        info_1 = ()

        action_2 = tf.constant([4], dtype=tf.int32)
        state_2 = [tf.constant([[-1.77655810e-08, -1.33685605e-24,  6.38598621e-01,
         7.61594176e-01, -3.75145068e-21,  9.86972850e-08,
        -1.48689061e-09, -1.74273446e-05, -2.63282836e-08,
         5.92726886e-01,  7.61565208e-01, -7.24647880e-01,
        -6.87594712e-01,  6.96363037e-11, -7.61572182e-01,
         2.09605772e-04, -7.32492089e-01,  3.36869410e-03,
         7.57435143e-01, -1.17230536e-06, -7.59913087e-01,
         5.44956933e-08,  2.06958025e-22, -2.87071002e-31,
         4.41342050e-18,  1.17024053e-13,  4.04977353e-20,
        -1.92512441e-15, -5.86423650e-03, -2.27814412e-09,
         2.63357260e-16,  3.35348815e-01, -7.61594176e-01,
        -2.50039260e-08, -5.32503832e-16, -1.46993140e-10,
         7.20538935e-07, -3.18827801e-07, -2.17475560e-12,
         1.76009803e-06]], dtype=tf.float32), tf.constant([[-1.7765588e-08, -2.6623662e-12,  1.0000000e+00,  1.0000000e+00,
        -9.9732494e-01,  2.6592757e-03, -1.0000000e+00, -3.3069274e-04,
        -2.8651908e-08,  6.8185931e-01,  1.0000000e+00, -9.1762531e-01,
        -1.0000000e+00,  9.9946231e-01, -9.9998695e-01,  9.4375189e-04,
        -9.3408352e-01,  1.0000000e+00,  9.9999887e-01, -9.3397319e-01,
        -9.9600911e-01,  1.0000000e+00,  2.0705544e-22, -4.2476038e-28,
         6.8120704e-10,  9.9999928e-01,  6.9739550e-02, -1.9255293e-15,
        -1.0000000e+00, -9.7590691e-01,  2.6335726e-16,  3.4918180e-01,
        -1.0000000e+00, -7.5932667e-07, -1.0944684e-08, -1.0000000e+00,
         7.2053894e-07, -3.1882780e-07, -8.7965959e-01,  1.7600980e-06]], dtype=tf.float32)]
        info_2 = ()

        action_3 = tf.constant([1], dtype=tf.int32)
        state_3 = [tf.constant([[-2.5365379e-10, -2.7658111e-30,  6.7390627e-01,  7.6159418e-01,
        -5.4917760e-26,  2.1259032e-09, -1.2548714e-11, -2.3801560e-06,
        -4.3676443e-10,  6.5463543e-01,  7.6159161e-01, -7.4061054e-01,
        -7.1625602e-01,  2.8363531e-13, -7.6159251e-01,  3.1447089e-05,
        -7.4597633e-01,  9.3062234e-04,  7.6038516e-01, -4.9777714e-08,
        -7.6121175e-01,  1.0841432e-09,  1.6877032e-27, -1.5360007e-38,
         3.2552203e-22,  1.0410070e-16,  1.0290848e-24, -6.0375649e-19,
        -1.8539239e-03, -2.1542436e-11,  5.1421906e-20,  3.5863534e-01,
        -7.6159418e-01, -1.4916640e-09, -1.2295118e-19, -7.1505285e-13,
         2.4842551e-08, -9.0529264e-09, -4.0635681e-15,  7.5063163e-08]], dtype=tf.float32), tf.constant([[-2.5365379e-10, -4.6707197e-15,  1.0000000e+00,  1.0000000e+00,
        -9.9934542e-01,  6.4990827e-04, -1.0000000e+00, -8.7498018e-05,
        -4.5839049e-10,  7.8336757e-01,  1.0000000e+00, -9.5186603e-01,
        -1.0000000e+00,  9.9992406e-01, -9.9999911e-01,  1.7989485e-04,
        -9.6382076e-01,  1.0000000e+00,  1.0000000e+00, -9.6374267e-01,
        -9.9908996e-01,  1.0000000e+00,  1.6878313e-27, -1.2913221e-34,
         4.4743163e-12,  1.0000000e+00,  3.8883656e-02, -6.0377340e-19,
        -1.0000000e+00, -9.9149644e-01,  5.1421906e-20,  3.7538844e-01,
        -1.0000000e+00, -9.9451256e-08, -1.3924421e-10, -1.0000000e+00,
         2.4842551e-08, -9.0529264e-09, -9.2149472e-01,  7.5063163e-08]], dtype=tf.float32)]
        info_3 = ()

        policy_step_1 = tfa.trajectories.policy_step.PolicyStep(action=action_1, state=state_1, info=info_1)
        policy_step_2 = tfa.trajectories.policy_step.PolicyStep(action=action_2, state=state_2, info=info_2)
        policy_step_3 = tfa.trajectories.policy_step.PolicyStep(action=action_3, state=state_3, info=info_3)
        policy_steps = [policy_step_1, policy_step_2, policy_step_3]
        return policy_steps
    
    def _policy_step_limits_from_set(self, policy_steps = None):
        min_limit = policy_steps[0].state
        max_limit = policy_steps[0].state
        for policy_step in policy_steps:
            min_limit = tf.minimum(min_limit, policy_step.state)
            max_limit = tf.maximum(max_limit, policy_step.state)
        return min_limit, max_limit
    
    def generate_policy_step_states_limits(self):
        time_step = self.create_timestep(self.tf_environment.nr_actions)
        actions = []
        for observation in self.unique_observations:
            time_step.observation['obs'] = tf.constant([[observation]], dtype=tf.int32)
            action = self._model.agent.policy.action(time_step)
            for i in range(10):
                action = self._model.agent.policy.action(time_step, action.state)
                actions.append(action)
            action = self._model.agent.policy.action(time_step)
            actions.append(action)
        limits_min, limits_max = self._policy_step_limits_from_set(actions)
        return limits_min, limits_max
    
    def monte_carlo_evaluation(self, time_step, nr_episodes, policy_limits_min=[0], policy_limits_max=[1]):
        unique_actions = np.empty(shape=(0,), dtype=np.int32)
        action = self._model.agent.policy.action(time_step)
        for _ in range(nr_episodes):
            for state in action.state:
                state = tf.random.uniform(shape=state.shape, 
                                                    minval=policy_limits_min, 
                                                    maxval=policy_limits_max, 
                                                    dtype=tf.float32)
            uniqor = self._model.agent.policy.action(time_step, action.state)
            unique_actions = np.unique(np.append(unique_actions, uniqor.action))
        return unique_actions
    
    def make_actions_printable(self, actions):
        action_numbers = ""
        action_keywords = ""
        for action in actions:
            action_numbers += f"{', ' if len(action_numbers) > 0 else ''}{action}"
            action_keywords += f"{', ' if len(action_keywords) > 0 else ''}{self.tf_environment.act_keywords[action]}"
        return action_numbers, action_keywords
    
    
    def evaluate_model(self, file, nr_episodes=10, method="monte_carlo"):
        time_step = self.create_timestep(self.tf_environment.nr_actions)
        self.unique_observations = np.unique(self.storm_model.observations)
        limits_min, limits_max = self.generate_policy_step_states_limits()
        for observation in self.unique_observations:
            time_step.observation['obs'] = tf.constant([[observation]], dtype=tf.int32)
            if method == "monte_carlo":
                action = self._model.agent.policy.action(time_step)
                actions = self.monte_carlo_evaluation(time_step, nr_episodes, limits_min, limits_max)
                acts, act_keys = self.make_actions_printable(actions)
                print(f"Observation: {observation}. Actions: {acts}. " +
                      f"Action Keywords: {act_keys}")
                file.write(f"Observation: {observation}. Action: {acts}. " +
                           f"Action Keyword: {act_keys}\n")
            else:
                print("No method specified.")
            # print(action)
    
    def evaluate_model_to_file(self, path_to_file):
        with open(path_to_file, "w") as file:
            self.evaluate_model(file)



if __name__ == "__main__":
    interface = RLInterface()
    interface.create_model()
    actions = interface.tf_environment._simulator.available_actions()
    interface.evaluate_model_to_file("obs_actions.txt")
    # for i in range(interface.storm_model.nr_observations):
    #     print(interface.storm_model.get_observation_labels(i))
    