# File: rl_main.py
# Description: Main for Reinforcement Learning Approach. If you want to train agents,
#              you can do it here, or you can use --reinforcement-learning option within PAYNT
# Author: David Hudak
# Login: xhudak03

from agents.father_agent import FatherAgent
from rl_src.agents.policies.parallel_fsc_policy import FSC_Policy, FSC
from interpreters.tracing_interpret import TracingInterpret
from interpreters.model_free_interpret import ModelFreeInterpret, ModelInfo

from agents.recurrent_ppo_agent import Recurrent_PPO_agent
from agents.recurrent_ddqn_agent import Recurrent_DDQN_agent
from agents.recurrent_dqn_agent import Recurrent_DQN_agent
from agents.ppo_with_qvalues_fsc import PPO_with_QValues_FSC
from agents.periodic_fsc_neural_ppo import Periodic_FSC_Neural_PPO

from environment import tf_py_environment
from rl_src.tools.saving_tools import save_dictionaries, save_statistics_to_new_json
from tools.evaluators import *
from environment.environment_wrapper import *
from environment.environment_wrapper_vec import *
from environment.pomdp_builder import *
from tools.args_emulator import ArgsEmulator, ReplayBufferOptions
from tools.weight_initialization import WeightInitializationMethods

import tensorflow as tf
import sys
import os

import logging

logger = logging.getLogger(__name__)

logging.getLogger('tensorflow').setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

sys.path.append("../")

import jax

tf.autograph.set_verbosity(0)


class ExperimentInterface:
    def __init__(self, args: ArgsEmulator = None, pomdp_model=None, agent=None):
        if args is None:
            self.args = ArgsEmulator()
        else:
            self.args = args
        self.pomdp_model = pomdp_model
        self.agent = agent

    def get_args(self) -> ArgsEmulator:
        return self.args

    def asserts(self):
        if self.args.prism_model and not self.args.prism_properties:
            raise ValueError("Prism model is set but Prism properties are not")
        if self.args.paynt_fsc_imitation and not self.args.paynt_fsc_json:
            raise ValueError(
                "Paynt imitation is set but there is not selected any JSON FSC file.")

    def initialize_prism_model(self):
        properties = parse_properties(self.args.prism_properties)
        pomdp_args = POMDP_arguments(
            self.args.prism_model, properties, self.args.constants)
        return POMDP_builder.build_model(pomdp_args)

    def run_agent(self):
        num_steps = 10
        for _ in range(num_steps):
            time_step = self.tf_environment._reset()
            is_last = time_step.is_last()
            while not is_last:
                action_step = self.agent.policy(time_step)
                next_time_step = self.tf_environment.step(action_step.action)
                time_step = next_time_step
                is_last = time_step.is_last()

    def initialize_environment(self, args: ArgsEmulator = None, pomdp_model=None):
        if pomdp_model is None:
            self.pomdp_model = self.initialize_prism_model()
        else:
            self.pomdp_model = pomdp_model
        logger.info("Model initialized")
        if self.args.replay_buffer_option == ReplayBufferOptions.ORIGINAL_OFF_POLICY or not self.args.vectorized_envs_flag:
            num_envs = 1
            self.args.num_environments = 1
        else:
            num_envs = self.args.num_environments
        if self.args.vectorized_envs_flag:
            environment = Environment_Wrapper_Vec(
                self.pomdp_model, args, num_envs=num_envs)
        else:
            environment = Environment_Wrapper(self.pomdp_model, args)
        # self.environment = Environment_Wrapper_Vec(self.pomdp_model, self.args, num_envs=num_envs)
        tf_environment = tf_py_environment.TFPyEnvironment(
            environment, check_dims=True)
        # self.tf_environment_orig = tf_py_environment.TFPyEnvironment(self.environment_orig)
        logger.info("Environment initialized")
        return environment, tf_environment

    def select_agent_type(self, learning_method=None, qvalues_table=None, action_labels_at_observation=None,
                          pre_training_dqn: bool = False) -> FatherAgent:
        """Selects the agent type based on the learning method and encoding method in self.args. The agent is saved to the self.agent variable.

        Args:
            learning_method (str, optional): The learning method. If set, the learning method is used instead of the one from the args object. Defaults to None.
            qvalues_table (dict, optional): The Q-values table created by the product of POMDPxFSC. Defaults to None.
        Raises:
            ValueError: If the learning method is not recognized or implemented yet."""
        if learning_method is None:
            learning_method = self.args.learning_method
        agent_folder = f"./trained_agents/{self.args.agent_name}_{self.args.learning_method}_{self.args.encoding_method}"
        if learning_method == "DQN":
            agent = Recurrent_DQN_agent(
                self.environment, self.tf_environment, self.args, load=self.args.load_agent, agent_folder=agent_folder,
                single_value_qnet=pre_training_dqn)
        elif learning_method == "DDQN":
            agent = Recurrent_DDQN_agent(
                self.environment, self.tf_environment, self.args, load=self.args.load_agent, agent_folder=agent_folder)
        elif learning_method == "PPO":
            agent = Recurrent_PPO_agent(
                self.environment, self.tf_environment, self.args, load=self.args.load_agent, agent_folder=agent_folder)
        elif learning_method == "Stochastic_PPO":
            self.args.prefer_stochastic = True
            agent = Recurrent_PPO_agent(
                self.environment, self.tf_environment, self.args, load=self.args.load_agent, agent_folder=agent_folder)
        else:
            raise ValueError(
                "Learning method not recognized or implemented yet.")
        return agent

    def initialize_agent(self, qvalues_table=None, action_labels_at_observation=None, learning_method=None,
                         pre_training_dqn: bool = False) -> FatherAgent:
        """Initializes the agent. The agent is initialized based on the learning method and encoding method. The agent is saved to the self.agent variable.
        It is important to have previously initialized self.environment, self.tf_environment and self.args.

        returns:
            FatherAgent: The initialized agent.
        """
        agent = self.select_agent_type(
            qvalues_table=qvalues_table, action_labels_at_observation=action_labels_at_observation,
            learning_method=learning_method, pre_training_dqn=pre_training_dqn)
        if self.args.restart_weights > 0:
            agent = WeightInitializationMethods.select_best_starting_weights(
                agent, self.args)
        return agent

    def initialize_fsc_agent(self):
        with open("FSC_experimental.json", "r") as f:
            fsc_json = json.load(f)
        fsc = FSC.from_json(fsc_json)
        action_keywords = self.environment.action_keywords
        policy = FSC_Policy(self.tf_environment, fsc,
                            tf_action_keywords=action_keywords)
        return policy

    def evaluate_random_policy(self):
        """Evaluates the random policy. The result is saved to the self.agent.evaluation_result object."""
        agent_folder = f"./trained_agents/{self.args.agent_name}_{self.args.learning_method}_{self.args.encoding_method}"
        self.agent = FatherAgent(
            self.environment, self.tf_environment, self.args, agent_folder=agent_folder)

        self.agent.evaluate_agent(
            False, vectorized=self.args.vectorized_envs_flag)

        results = {}
        if self.args.perform_interpretation:
            interpret = TracingInterpret(self.environment, self.tf_environment,
                                         self.args.encoding_method, self.environment._possible_observations)
            for refusing in [True, False]:
                result = interpret.get_dictionary(self.agent, refusing)
                if refusing:
                    results["best_with_refusing"] = result
                    results["last_with_refusing"] = result
                else:
                    results["best_without_refusing"] = result
                    results["last_without_refusing"] = result
        return results

    def tracing_interpretation(self, with_refusing=None):
        interpret = TracingInterpret(self.environment, self.tf_environment,
                                     self.args.encoding_method)
        for quality in ["last", "best"]:
            logger.info(f"Interpreting agent with {quality} quality")
            if with_refusing == None:
                result = {}
                self.agent.load_agent(quality == "best")
                result[f"{quality}_with_refusing"] = interpret.get_dictionary(
                    self.agent, with_refusing=True, vectorized=self.args.vectorized_envs_flag)
                result[f"{quality}_without_refusing"] = interpret.get_dictionary(
                    self.agent, with_refusing=False, vectorized=self.args.vectorized_envs_flag)
            else:
                result = interpret.get_dictionary(self.agent, with_refusing)
        return result

    def perform_experiment(self, with_refusing=False):
        """Performs the experiment. The experiment is performed based on the arguments in self.args. The result is saved to the self.agent variable.
        Additional experimental data can be found in the self.agent.evaluation_result variable.

        Returns:
            dict: The result of the experiment.
        """
        try:
            self.asserts()
        except ValueError as e:
            logger.error(e)
            return
        self.environment, self.tf_environment = self.initialize_environment(
            self.args)
        if self.args.evaluate_random_policy:  # Evaluate random policy
            return self.evaluate_random_policy()

        self.agent = self.initialize_agent()
        self.agent.train_agent(self.args.nr_runs, vectorized=self.args.vectorized_envs_flag,
                               replay_buffer_option=self.args.replay_buffer_option)
        self.agent.save_agent()
        result = {}
        if self.args.perform_interpretation:
            logger.info("Training finished")
            if self.args.interpretation_method == "Tracing":
                result = self.tracing_interpretation(with_refusing)
            else:
                raise ValueError(
                    "Interpretation method not recognized or implemented yet.")
        return result

    def __del__(self):
        if hasattr(self, "tf_environment") and self.tf_environment is not None:
            try:
                self.tf_environment.close()

            except Exception as e:
                pass
        if hasattr(self, "agent") and self.agent is not None:
            self.agent.save_agent()
        jax.clear_caches()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initializer = ExperimentInterface()
    args = ArgsEmulator()
    result = initializer.perform_experiment(args.with_refusing)
    if args.with_refusing is None:
        save_dictionaries(args.experiment_directory, args.agent_name,
                          args.learning_method, "best_with_refusing", result["best_with_refusing"][0], result["best_with_refusing"][1], result["best_with_refusing"][2])

        save_dictionaries(args.experiment_directory, args.agent_name,
                          args.learning_method, "last_with_refusing",
                          result["last_with_refusing"][0], result["last_with_refusing"][1],
                          result["last_with_refusing"][2])
        save_statistics_to_new_json(args.experiment_directory, args.agent_name, args.learning_method,
                                    initializer.agent.evaluation_result, args)
        save_dictionaries(args.experiment_directory, args.agent_name,
                          args.learning_method, "best_without_refusing", result[
                              "best_without_refusing"][0],
                          result["best_without_refusing"][1], result["best_without_refusing"][2])
        save_dictionaries(args.experiment_directory, args.agent_name,
                          args.learning_method, "last_without_refusing", result[
                              "last_without_refusing"][0],
                          result["last_without_refusing"][1], result["last_without_refusing"][2])
    else:
        save_dictionaries(args.experiment_directory, args.agent_name,
                          args.learning_method, args.with_refusing, result[0], result[1], result[2])
        save_statistics_to_new_json(args.experiment_directory, args.agent_name, args.learning_method,
                                    initializer.agent.evaluation_result, args)
