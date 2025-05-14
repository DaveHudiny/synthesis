# Author: Martin Tappler, reimplemented by David Hudák

import aalpy
import numpy as np

from tf_agents.policies import TFPolicy
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories import trajectory
from tf_agents.policies import PyTFEagerPolicy

from environment.tf_py_environment import TFPyEnvironment
from environment.environment_wrapper_vec import EnvironmentWrapperVec
from tools.args_emulator import ArgsEmulator
from interpreters.extracted_fsc.table_based_policy import TableBasedPolicy
from tools.evaluators import evaluate_policy_in_model
from agents.recurrent_ppo_agent import Recurrent_PPO_agent

from interpreters.aalpy_extraction.mealy_automata_learner import MealyAutomataLearner

from tests.general_test_tools import *


class AALpyExtractor:
    """
    AALpyExtractor is a class that uses the AALpy library to extract a Finite State Controller (FSC) from a given policy.
    """

    def __init__(self, env : EnvironmentWrapperVec, args : ArgsEmulator, num_steps : int = 1000, num_envs : int = 256):
        """
        Initializes the AALpyExtractor with a given policy, environment, and arguments.

        Args:
            env: The environment in which the policy operates.
            args: Additional arguments for the extraction process.
        """

        self.env = env
        self.tf_env = TFPyEnvironment(env)
        self.args = args
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.counter = 0

    def initialize_storage(self, policy: TFPolicy, num_steps : int, num_envs : int) -> TFUniformReplayBuffer:
        """
        Initializes a replay buffer for storing the sampled data.

        Args:
            policy (TFPolicy): The policy to be used for sampling.

        Returns:
            TFUniformReplayBuffer: The initialized replay buffer.
        """
        collect_data_spec = policy.collect_data_spec
        # Remove all parts of observation excluding the "integer" part
        collect_data_spec = collect_data_spec._replace(
            observation=collect_data_spec.observation['integer']
        )
        return TFUniformReplayBuffer(
            data_spec=collect_data_spec,
            batch_size=num_envs,
            max_length=num_steps + 100 # +100 is because the step driver is not deterministic and can add more steps
        )
    
    def get_add_batch(self, replay_buffer: TFUniformReplayBuffer):
        """
        Returns the modified add_batch method of the replay buffer.

        Args:
            replay_buffer (TFUniformReplayBuffer): The replay buffer to be used.

        Returns:
            callable: The add_batch method of the replay buffer.
        """
        return lambda trajectory: replay_buffer.add_batch(trajectory._replace(
            observation=trajectory.observation['integer']
        ))        

    def initialize_vec_driver(self, policy: TFPolicy, observer : callable, num_steps : int, num_envs : int) -> DynamicStepDriver:
        """
        Initializes a vectorized driver for sampling data from the environment.

        Args:
            policy (TFPolicy): The policy to be used for sampling.

        Returns:
            DynamicStepDriver: The initialized driver.
        """
        num_overall_steps = (num_steps) * num_envs
        eager_policy = PyTFEagerPolicy(
            policy,
            use_tf_function=True,
            batch_time_steps=False
        )
        return DynamicStepDriver(
            env=self.tf_env,
            policy=eager_policy,
            num_steps=num_overall_steps,
            observers=[observer],
        )
        
    def get_add_counter(self):
        """
        Increments the counter by 1.

        Args:
            **kwargs: Additional arguments.
        """
        def __add_counter(item):
            self.counter += 1
            
        return __add_counter

    def sample_data(self, policy: TFPolicy, num_envs, num_steps):
        """
        Samples data from the environment using the given policy.

        Args:
            policy (TFPolicy): The policy to be used for sampling.
            num_envs (int): The number of environments to sample from.
            num_steps (int): The number of steps to sample.

        Returns:
            tuple: A tuple containing the sampled observations, actions, and rewards.
        """
        orig_num_envs = self.env.num_envs
        self.env.set_num_envs(num_envs)
        

        replay_buffer = self.initialize_storage(policy, num_steps, num_envs)
        driver = self.initialize_vec_driver(policy, self.get_add_batch(replay_buffer), num_steps, num_envs)
        self.tf_env.reset()
        driver.run()
        self.env.set_num_envs(orig_num_envs)
        return replay_buffer.gather_all()

    def extract_mealy_machine(self, policy: TFPolicy) -> tuple[aalpy.MealyMachine, list[str]]:
        """
        Extracts a Mealy machine from the given policy using the AALpy library.
        """
        # Sample data from the environment using the given policy
        trajectories = self.sample_data(policy, self.num_envs, self.num_steps)
        # Convert the sampled data into a list of traces
        return MealyAutomataLearner.extract_mealy_machine(trajectories), self.env.action_keywords
        

    def create_action_function(self, model : aalpy.MealyMachine, nr_model_states, nr_observations):
        """
        Creates an action function for the given Mealy machine.

        Args:
            model (aalpy.MealyMachine): The Mealy machine to create the action function for.
            nr_model_states (int): The number of states in the model.
            nr_observations (int): The number of observations in the model.

        Returns:
            np.ndarray: The action function as a numpy array.
        """
        action_function = np.zeros((nr_model_states, nr_observations), dtype=np.int32)
        for i in range(nr_model_states):
            for j in range(nr_observations):
                if f"[{j}]" in model.states[i].output_fun:
                    action_function[i][j] = model.states[i].output_fun[f"[{j}]"] if model.states[i].output_fun[f"[{j}]"] != "epsilon" else 0
                else:
                    action_function[i][j] = 0
        return action_function
    
    def create_update_function(self, model : aalpy.MealyMachine, nr_model_states, nr_observations):
        """
        Creates an update function for the given Mealy machine.

        Args:
            model (aalpy.MealyMachine): The Mealy machine to create the update function for.
            nr_model_states (int): The number of states in the model.
            nr_observations (int): The number of observations in the model.

        Returns:
            np.ndarray: The update function as a numpy array.
        """
        update_function = np.zeros((nr_model_states, nr_observations), dtype=np.int32)
        state_ids = [state.state_id for state in model.states]
        state_id_to_index = {state_id: i for i, state_id in enumerate(state_ids)}
        for i in range(nr_model_states):
            for j in range(nr_observations):
                if f"[{j}]" in model.states[i].transitions:
                    update_function[i][j] = state_id_to_index[model.states[i].transitions[f"[{j}]"].state_id]
                else:
                    update_function[i][j] = 0
        return update_function


    def extract_fsc(self, policy: TFPolicy) -> TableBasedPolicy:
        """
        Extracts a Finite State Controller (FSC) from the given policy using the AALpy library.

        Args:
            policy (TFPolicy): The policy to be extracted.

        Returns:
            TableBasedPolicy: The extracted FSC as a table-based policy.
        """
        mealy_machine, action_labels = self.extract_mealy_machine(policy)
        nr_model_states = len(mealy_machine.states)
        nr_observations = self.env.vectorized_simulator.simulator.observation_by_ids.shape[0]
        action_function = self.create_action_function(mealy_machine, nr_model_states, nr_observations)
        update_function = self.create_update_function(mealy_machine, nr_model_states, nr_observations)
        table_based_policy = TableBasedPolicy(
            original_policy=policy,
            action_function=action_function,
            update_function=update_function,
            action_keywords=action_labels
        )
        self.evaluate_policy(table_based_policy)
        return table_based_policy
    
    def evaluate_policy(self, policy: TableBasedPolicy) -> float:
        """
        Evaluates the given policy in the environment.

        Args:
            policy (TableBasedPolicy): The policy to be evaluated.

        Returns:
            float: The average reward obtained by the policy.
        """
        return evaluate_policy_in_model(self.env, policy, self.args.nr_runs)
    
if __name__ == "__main__":
    # Example usage
    prism_model = "models/intercept/sketch.templ"
    prism_properties = "models/intercept/sketch.props"

    args = init_args(prism_model, prism_properties)
    env, tf_env = init_environment(args)
    agent = Recurrent_PPO_agent(env, tf_env, args)
    # agent.train_agent(50)
    policy = agent.get_policy(eager=False)
    
    extractor = AALpyExtractor(env, args)
    fsc = extractor.extract_fsc(policy)
    # Evaluate the extracted FSC
    evaluation_result = extractor.evaluate_policy(fsc)
    

        