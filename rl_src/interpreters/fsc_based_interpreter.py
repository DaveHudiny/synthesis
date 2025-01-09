from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.policy_step import PolicyStep

from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy

from rl_src.environment.environment_wrapper_vec import Environment_Wrapper_Vec

import numpy as np
import tensorflow as tf

import logging
import pickle as pkl

from tools.args_emulator import ArgsEmulator

import os

logger = logging.getLogger(__name__)

def create_fake_timestep(observation_triplet):
    return TimeStep(
        step_type=tf.constant([0], dtype=tf.int32),
        reward=tf.constant([0], dtype=tf.float32),
        discount=tf.constant([1], dtype=tf.float32),
        observation=observation_triplet
    )

def save_extracted_fsc(observation_to_action_table, observation_to_update_table, action_labels, memory_size, observation_size, args : ArgsEmulator, model_name,
                       percentage_of_misses=None):
    saved_dict = {
        "observation_to_action_table": observation_to_action_table,
        "observation_to_update_table": observation_to_update_table,
        "action_labels": action_labels,
        "memory_size": memory_size,
        "observation_size": observation_size,
        "percentage_of_misses": percentage_of_misses
    }
    name = args.name_of_experiment + "/" + args.agent_name + "_" + "_extracted_fsc.pkl"
    # Check if the folder exists
    if not os.path.exists(args.name_of_experiment):
        os.makedirs(args.name_of_experiment)
    # Check if the file exists
    if os.path.exists(name):
        logger.info("File %s already exists. Finding a new index.", name)
        index = 1
        while os.path.exists(name):
            name = args.name_of_experiment + "/" + args.agent_name + "_" + "_extracted_fsc_" + str(index) + ".pkl"
            index += 1

    with open(name, "wb") as f:
        logger.info("Saving extracted FSC to %s", name)
        pkl.dump(saved_dict, f)

def construct_table_observation_action_memory( agent_policy : TFPolicy, environment : Environment_Wrapper_Vec):
    """Constructs a table with observations, actions and memory values.

    Args:
        agent_policy (TFPolicy): The agent's policy.

    Returns:
        dict: The dictionary with observations and memory tuples as keys and actions as values.
    """
    state_to_observations = np.array(environment.stormpy_model.observations)
    all_observations = range(np.unique(state_to_observations).shape[0])
    model_memory_size = environment.model_memory_size if environment.model_memory_size > 0 else 1
    no_memory = True if environment.model_memory_size == 0 else False
    observation_to_action_table = np.zeros((model_memory_size, len(all_observations), )) 
    observation_to_update_table = np.zeros((model_memory_size, len(all_observations), ))
    number_of_misses = 0

    for integer_observation in all_observations:
        for memory in range(model_memory_size):
            # Find some of the states that correspond to the integer observation
            state = np.where(state_to_observations == integer_observation)[0][0]
            observation_triplet = environment.encode_observation(integer_observation, memory, state)
            mask = observation_triplet["mask"]
            time_step = create_fake_timestep(observation_triplet)
            played_action = agent_policy.action(time_step=time_step)
            if no_memory:
                action = played_action.action
                update = 0
            else:
                action = played_action.action["simulator_action"]
                update = played_action.action["memory_update"]
            if not mask[action].numpy():
                number_of_misses += 1
            if mask.numpy().sum() == 1:
                # Only a single action is legal -- pick it.
                action = mask.numpy().argmax()
            observation_to_action_table[memory][integer_observation] = action
            observation_to_update_table[memory][integer_observation] = update
    
    logger.info("Number of misses: %d", number_of_misses)
    logger.info("Percentage of misses: %f", number_of_misses / (len(all_observations) * model_memory_size))
    save_extracted_fsc(observation_to_action_table, observation_to_update_table, environment.act_to_keywords, model_memory_size, len(all_observations), environment.args, environment.args.agent_name,
                       percentage_of_misses=number_of_misses / (len(all_observations) * model_memory_size))
    
    return observation_to_action_table, observation_to_update_table

class ExtractedFSCPolicy(TFPolicy):
    def __init__(self, agent_policy : TFPolicy, environment : Environment_Wrapper_Vec, tf_environment : TFPyEnvironment, args, model = ""):
        eager = PyTFEagerPolicy(agent_policy, use_tf_function=True)
        self.observation_to_action_table, self.observation_to_update_table = construct_table_observation_action_memory(eager, environment)
        self.tf_observation_to_action_table = tf.constant(self.observation_to_action_table, dtype=tf.int32)
        self.tf_observation_to_update_table = tf.constant(self.observation_to_update_table, dtype=tf.int32)
        self.action_labels = environment.act_to_keywords
        self.memory_size = environment.model_memory_size

        self.observation_size = len(environment.stormpy_model.observations)
        self.args = args
        super(ExtractedFSCPolicy, self).__init__(tf_environment.time_step_spec(), tf_environment.action_spec())

    def _get_initial_state(self, batch_size):
        return ()
    
    def get_single_action(self, observation, memory):
        return self.observation_to_action_table[memory, observation]
    
    def get_single_update(self, observation, memory):
        return self.observation_to_update_table[memory, observation]
    
    def set_single_action(self, observation, memory, action):
        self.observation_to_action_table[memory, observation] = action

    def set_single_update(self, observation, memory, update):
        self.observation_to_update_table[memory, observation] = update

    def recompile_tf_tables(self):
        self.tf_observation_to_action_table = tf.constant(self.observation_to_action_table, dtype=tf.int32)
        self.tf_observation_to_update_table = tf.constant(self.observation_to_update_table, dtype=tf.int32)
    
    def _action(self, time_step : TimeStep, policy_state : PolicyStep, seed):
        observation = time_step.observation["integer"]
        if self.memory_size == 0:
            memory = tf.zeros(shape=(time_step.observation["observation"].shape[0], 1), dtype=tf.int32)
        else:
            memory = tf.cast(time_step.observation["observation"][:, -1], dtype=tf.int32)
            memory = tf.reshape(memory, (-1, 1))
        
        indices = tf.concat([memory, observation], axis=1)
        action = tf.gather_nd(self.tf_observation_to_action_table, indices)
        update = tf.gather_nd(self.tf_observation_to_update_table, indices)
        # action = self.observation_to_action_table[memory, observation]
        # update = self.observation_to_update_table[memory, observation]
        if self.memory_size == 0:
            action_dict = action
        else:
            action_dict = {
                "simulator_action": action,
                "memory_update": update
            }

        return PolicyStep(action_dict, policy_state)



    