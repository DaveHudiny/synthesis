from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.policy_step import PolicyStep

from tf_agents.specs.tensor_spec import TensorSpec

from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy

from rl_src.environment.environment_wrapper_vec import Environment_Wrapper_Vec

from rl_src.agents.policies.policy_mask_wrapper import Policy_Mask_Wrapper
# from rl_src.agents.father_agent import FatherAgent

# from rl_src.interpreters.tracing_interpret import TracingInterpret

import numpy as np
import tensorflow as tf

import logging
import pickle as pkl

from tools.args_emulator import ArgsEmulator

import os

logger = logging.getLogger(__name__)


def create_fake_timestep(observation_triplet):
    return TimeStep(
        step_type=tf.constant(0, dtype=tf.int32),
        reward=tf.constant(0, dtype=tf.float32),
        discount=tf.constant(1, dtype=tf.float32),
        observation=observation_triplet
    )


def save_extracted_fsc(observation_to_action_table, observation_to_update_table, action_labels, memory_size, observation_size, args: ArgsEmulator, model_name,
                       percentage_of_misses=None):
    saved_dict = {
        "observation_to_action_table": observation_to_action_table,
        "observation_to_update_table": observation_to_update_table,
        "action_labels": action_labels,
        "memory_size": memory_size,
        "observation_size": observation_size,
        "percentage_of_misses": percentage_of_misses
    }
    name = args.name_of_experiment + "/" + \
        args.agent_name + "_" + "_extracted_fsc.pkl"
    # Check if the folder exists
    if not os.path.exists(args.name_of_experiment):
        os.makedirs(args.name_of_experiment)
    # Check if the file exists
    if os.path.exists(name):
        logger.info("File %s already exists. Finding a new index.", name)
        index = 1
        while os.path.exists(name):
            name = args.name_of_experiment + "/" + args.agent_name + \
                "_" + "_extracted_fsc_" + str(index) + ".pkl"
            index += 1

    with open(name, "wb") as f:
        logger.info("Saving extracted FSC to %s", name)
        pkl.dump(saved_dict, f)


def construct_table_observation_action_memory(agent_policy: TFPolicy, environment: Environment_Wrapper_Vec, memory_size=0) -> tuple[np.ndarray, np.ndarray]:
    """Constructs a table with observations, actions and memory values.

    Args:
        agent_policy (TFPolicy): The agent's policy.

    Returns:
        dict: The dictionary with observations and memory tuples as keys and actions as values.
    """
    state_to_observations = np.array(environment.stormpy_model.observations)
    all_observations = range(np.unique(state_to_observations).shape[0])
    model_memory_size = memory_size if memory_size > 0 else 1
    no_memory = True if memory_size == 0 else False
    observation_to_action_table = np.zeros(
        (model_memory_size, len(all_observations), ))
    observation_to_update_table = np.zeros(
        (model_memory_size, len(all_observations), ))
    number_of_misses = 0

    for integer_observation in all_observations:
        for memory in range(model_memory_size):
            # Find some of the states that correspond to the integer observation
            state = np.where(state_to_observations ==
                             integer_observation)[0][0]
            observation_triplet = environment.encode_observation(
                integer_observation, memory, state)
            mask = observation_triplet["mask"]
            time_step = create_fake_timestep(observation_triplet)
            fake_policy_state = agent_policy.get_initial_state(1)
            played_action = agent_policy.action(
                time_step=time_step, policy_state=fake_policy_state)
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
    logger.info("Percentage of misses: %f", number_of_misses /
                (len(all_observations) * model_memory_size))
    # save_extracted_fsc(observation_to_action_table, observation_to_update_table, environment.act_to_keywords, 
    #                    model_memory_size, len(all_observations), environment.args, environment.args.agent_name,
    #                    percentage_of_misses=number_of_misses / (len(all_observations) * model_memory_size))

    return observation_to_action_table, observation_to_update_table


def construct_memoryless_table_w_entropy(agent_policy: Policy_Mask_Wrapper,
                                         environment: Environment_Wrapper_Vec,
                                         greedy: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constructs a table with observations and actions for memoryless policies with additional list of entropy values."""
    state_to_observations = np.array(environment.stormpy_model.observations)
    all_observations = range(np.unique(state_to_observations).shape[0])
    observation_to_action_table = np.zeros((len(all_observations), ))
    observation_to_entropy_table = np.zeros((len(all_observations), ))
    observation_to_update_table = np.zeros((len(all_observations), ))

    for integer_observation in all_observations:
        state = np.where(state_to_observations == integer_observation)[0][0]
        observation_triplet = environment.encode_observation(
            integer_observation, 0, state)
        time_step = create_fake_timestep(observation_triplet)
        fake_policy_state = agent_policy.get_initial_state(1)

        played_action = agent_policy.action(
            time_step=time_step, policy_state=fake_policy_state)
        logits = played_action.info["dist_params"]["logits"]
        # logits = tf.where(observation_triplet["mask"], logits, -np.inf)
        if not greedy:
            action = played_action.action
        else:
            action = tf.argmax(logits)
        observation_to_action_table[integer_observation] = action
        logits = played_action.info["dist_params"]["logits"]
        entropy = -tf.reduce_sum(tf.nn.softmax(logits)
                                 * (tf.nn.log_softmax(logits) / tf.math.log(2.0)), axis=-1)
        observation_to_entropy_table[integer_observation] = entropy
    return observation_to_action_table, observation_to_entropy_table, observation_to_update_table


class ExtractedFSCPolicy(TFPolicy):
    def __init__(self, time_step_spec, action_spec, policy_state_spec, observation_to_action_table = None, observation_to_update_table = None, labels = None, memory_size = 1, observation_size = 1):
        super(ExtractedFSCPolicy, self).__init__(
            time_step_spec, action_spec, policy_state_spec)
        if observation_to_action_table is not None:
            self.observation_to_action_table = tf.constant(observation_to_action_table, dtype=tf.int32)
        else:
            self.tf_observation_to_action_table = tf.zeros(
                shape=(memory_size, observation_size), dtype=tf.int32)
        if observation_to_update_table is not None:
            self.observation_to_update_table = tf.constant(observation_to_update_table, dtype=tf.int32)
        else:
            self.tf_observation_to_update_table = tf.zeros(
                shape=(memory_size, observation_size), dtype=tf.int32)
        self.model_memory_size = 0
        self.action_labels = labels

    def _action(self, time_step: TimeStep, policy_state: PolicyStep, seed):
        observation = time_step.observation["integer"]
        # if self.model_memory_size == 0:
        #     memory = policy_state
        # else:
        #     memory = tf.cast(
        #         time_step.observation["observation"][:, -1], dtype=tf.int32)
        #     memory = tf.reshape(memory, (-1, 1))
        memory = policy_state
        indices = tf.concat([memory, observation], axis=1)
        action = tf.gather_nd(self.tf_observation_to_action_table, indices)
        update = tf.gather_nd(self.tf_observation_to_update_table, indices)
        update = tf.reshape(update, (-1, 1))
        # action = self.observation_to_action_table[memory, observation]
        # update = self.observation_to_update_table[memory, observation]
        if self.model_memory_size == 0:
            action_dict = action
            policy_state = update
        else:
            action_dict = {
                "simulator_action": action,
                "memory_update": update
            }

        return PolicyStep(action_dict, policy_state)
    
    def _get_initial_state(self, batch_size):
        return tf.zeros((batch_size, 1), dtype=tf.int32)

class NaiveFSCPolicyExtraction(ExtractedFSCPolicy):
    def __init__(self, agent_policy: TFPolicy, environment: Environment_Wrapper_Vec, 
                 tf_environment: TFPyEnvironment, args, model="", entropy_extraction=False,
                 greedy: bool = False, max_memory_size = 10):
        eager = PyTFEagerPolicy(agent_policy, use_tf_function=True)
        self.state_to_observations = np.array(environment.stormpy_model.observations)
        self.all_observations = range(np.unique(self.state_to_observations).shape[0])
        self.minimum_logit = -1e30
        self.model_memory_size = environment.model_memory_size
        self.memory_size = environment.model_memory_size if environment.model_memory_size > 0 else 1
        self._init_action_and_memory_tables(agent_policy, environment)
        self.pre_computed_logits = {}
        if not entropy_extraction:
            self.observation_to_action_table, self.observation_to_update_table = construct_table_observation_action_memory(
                eager, environment)
            self.tf_observation_to_action_table = tf.constant(
                self.observation_to_action_table, dtype=tf.int32)
            self.tf_observation_to_update_table = tf.constant(
                self.observation_to_update_table, dtype=tf.int32)
        else:
            entropy_only = self.entropy_only_pass(eager, environment)
            self.observation_to_entropy_table = entropy_only
            self.update_tables_by_entropy_memory_estimation(entropy_only, max_memory_size = max_memory_size)
            self.fill_tables(eager, environment)
            self.recompile_tf_tables()

        if entropy_extraction:
            self.tf_observation_to_entropy_table = tf.constant(
                self.observation_to_entropy_table, dtype=tf.float32)
        self.action_labels = environment.act_to_keywords
        
        self.observation_size = environment.stormpy_model.nr_observations
        self.args = args
        # Policy state should contain a memory value
        if self.model_memory_size == 0:
            policy_state_spec = TensorSpec(shape=(), dtype=tf.int32)
        else:
            policy_state_spec = ()
        super(NaiveFSCPolicyExtraction, self).__init__(
            tf_environment.time_step_spec(), tf_environment.action_spec(), policy_state_spec=policy_state_spec)
        
    def _init_action_and_memory_tables(self, agent_policy: TFPolicy, environment: Environment_Wrapper_Vec):
        self.observation_to_action_table = np.zeros(
            (self.memory_size, len(self.all_observations), ))
        self.observation_to_update_table = np.zeros(
            (self.memory_size, len(self.all_observations), ))

    def entropy_only_pass(self, agent_policy: Policy_Mask_Wrapper, environment: Environment_Wrapper_Vec):
        observation_to_entropy_table = np.zeros((len(self.all_observations), ))
        self.pre_computed_logits = {}
        for integer_observation in self.all_observations:
            state = np.where(self.state_to_observations == integer_observation)[0][0]
            observation_triplet = environment.encode_observation(
                integer_observation, 0, state)
            time_step = create_fake_timestep(observation_triplet)
            fake_policy_state = agent_policy.get_initial_state(1)
            played_action = agent_policy.action(
                time_step=time_step, policy_state=fake_policy_state)
            logits = played_action.info["dist_params"]["logits"]
            logits = tf.where(observation_triplet["mask"], logits, self.minimum_logit)
            self.pre_computed_logits[integer_observation] = logits
            entropy = -tf.reduce_sum(tf.nn.softmax(logits)
                                    * (tf.nn.log_softmax(logits) / tf.math.log(2.0)), axis=-1)
            observation_to_entropy_table[integer_observation] = entropy
        return observation_to_entropy_table
    
    def compute_logits_for_observation(self, observation, agent_policy: Policy_Mask_Wrapper, environment: Environment_Wrapper_Vec):
        state = np.where(self.state_to_observations == observation)[0][0]
        observation_triplet = environment.encode_observation(
            observation, 0, state)
        time_step = create_fake_timestep(observation_triplet)
        fake_policy_state = agent_policy.get_initial_state(1)
        played_action = agent_policy.action(
            time_step=time_step, policy_state=fake_policy_state)
        logits = played_action.info["dist_params"]["logits"]
        logits = tf.where(observation_triplet["mask"], logits, self.minimum_logit)
        return logits

    def fill_tables(self, policy: Policy_Mask_Wrapper, environment: Environment_Wrapper_Vec):
        obs_shape = self.observation_to_action_table.shape[1]
        mem_shape = self.observation_to_action_table.shape[0]
        for obs in range(obs_shape):
            if obs in self.pre_computed_logits:
                logits = self.pre_computed_logits[obs]
            else:
                logits = self.compute_logits_for_observation(obs, policy, environment)
                self.pre_computed_logits[obs] = logits
            sorted_indices = tf.argsort(logits, direction='DESCENDING')
            for mem in range(mem_shape):
                if logits[sorted_indices[mem]] == self.minimum_logit:
                    # Select random action with logit higher than minimum value
                    non_inf_indices = np.argwhere(logits > self.minimum_logit)
                    if len(non_inf_indices) == 0:
                        action = 0
                    else:
                        non_inf_indices = non_inf_indices[0]
                        action = np.random.choice(non_inf_indices)
                else:
                    action = sorted_indices[mem]
                self.observation_to_action_table[mem, obs] = action
                self.observation_to_update_table[mem, obs] = 0
            

    def _get_initial_state(self, batch_size):
        return tf.zeros((batch_size, 1), dtype=tf.int32)

    def get_single_action(self, observation, memory):
        return self.observation_to_action_table[memory, observation]

    def get_single_update(self, observation, memory):
        return self.observation_to_update_table[memory, observation].astype(int)

    def set_single_action(self, observation, memory, action):
        self.observation_to_action_table[memory, observation] = action

    def set_single_update(self, observation, memory, update):
        self.observation_to_update_table[memory, observation] = update

    def get_memory_estimation_by_entropy(self, observation_to_entropy_table, capped_memory_size,  ceil : bool = True):
        if capped_memory_size == 0:
            return [0] * self.observation_size

        bit_entropies = observation_to_entropy_table
        memory_entropies = tf.pow(2.0, bit_entropies)
        clipped_memory_entropies = tf.clip_by_value(memory_entropies, 1, capped_memory_size)
        if ceil:
            memory_entropies = tf.math.ceil(clipped_memory_entropies).numpy()
        else:
            memory_entropies = tf.math.floor(clipped_memory_entropies).numpy()
        memory_entropies = memory_entropies.astype(int)
        return memory_entropies


    def update_tables_by_entropy_memory_estimation(self, observation_to_entropy_table, max_memory_size = 2): 
        memory_estimations = self.get_memory_estimation_by_entropy(observation_to_entropy_table, max_memory_size)
        maximum_memory = tf.reduce_max(memory_estimations).numpy()
        self.memory_size = maximum_memory
        if len(self.observation_to_action_table.shape) == 1:
            # Add dimension for memory
            self.observation_to_action_table = tf.expand_dims(self.observation_to_action_table, axis=0)
        if self.observation_to_action_table.shape[0] >= maximum_memory:
            return
        else:
            current_size = self.observation_to_action_table.shape[0]
            padding_ranks = tf.constant([[0, maximum_memory - current_size], [0, 0]], dtype=tf.int32)
            self.observation_to_action_table = tf.pad(
                self.observation_to_action_table, padding_ranks).numpy()
            self.observation_to_update_table = tf.pad(
                self.observation_to_update_table, padding_ranks).numpy()

    def recompile_tf_tables(self):
        self.tf_observation_to_action_table = tf.constant(
            self.observation_to_action_table, dtype=tf.int32)
        self.tf_observation_to_update_table = tf.constant(
            self.observation_to_update_table, dtype=tf.int32)
        
    @staticmethod    
    def geometric_legal_selection(options, multiplier = 4.0):
        options_range = np.arange(len(options)) + 1.0
        options_prob = 1.0 / (multiplier ** options_range)
        normalized_options_prob = options_prob / np.sum(options_prob)
        random_option = np.random.choice(options, p=normalized_options_prob) 
        return random_option
        
    def shuffle_memory(self):
        # goes through all observations and shuffles memory updates in observation_to_update_table
        logger.info("Shuffling memory")
        for obs in range(self.observation_size):
            options = np.arange(self.memory_size)
            for mem in range(self.memory_size):
                self.observation_to_update_table[mem, obs] = self.geometric_legal_selection(options)
                options = np.delete(options, np.where(options == self.observation_to_update_table[mem, obs]))
    

if __name__ == '__main__':
    # Test the ExtractedFSCPolicy class on a random Policy
    from rl_src.tests.general_test_tools import *
    from rl_src.agents.recurrent_ppo_agent import Recurrent_PPO_agent
    prism_path = "../models/mba/sketch.templ"
    properties_path = "../models/mba/sketch.props"
    args = init_args(prism_path=prism_path, properties_path=properties_path)
    env, tf_env = init_environment(args)
    agent_policy = Recurrent_PPO_agent(env, tf_env, args).wrapper
    extracted_fsc_policy = NaiveFSCPolicyExtraction(agent_policy, env, tf_env, args, entropy_extraction=True, greedy=True)
