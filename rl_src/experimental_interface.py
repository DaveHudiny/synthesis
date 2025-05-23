# File: rl_main.py
# Description: Main for Reinforcement Learning Approach. If you want to train agents,
#              you can do it here, or you can use --reinforcement-learning option within PAYNT
# Author: David Hudak
# Login: xhudak03

from agents.father_agent import FatherAgent
from tools.trajectory_buffer import TrajectoryBuffer
from tools.evaluation_results_class import EvaluationResults, log_evaluation_info
from rl_src.agents.policies.parallel_fsc_policy import FSC_Policy, FSC
from interpreters.tracing_interpret import TracingInterpret

from agents.recurrent_ppo_agent import Recurrent_PPO_agent
from agents.recurrent_ddqn_agent import Recurrent_DDQN_agent
from agents.recurrent_dqn_agent import Recurrent_DQN_agent

from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

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

    @staticmethod
    def initialize_prism_model(args : ArgsEmulator):
        properties = parse_properties(args.prism_properties)
        pomdp_args = POMDP_arguments(
            args.prism_model, properties, args.constants)
        return POMDP_builder.build_model(pomdp_args)

    def run_agent(self):
        num_steps = 10
        for _ in range(num_steps):
            time_step = self.tf_environment._reset()
            is_last = time_step.is_last()
            while not is_last:
                action_step = self.agent.action(time_step)
                next_time_step = self.tf_environment.step(action_step.action)
                time_step = next_time_step
                is_last = time_step.is_last()

    @staticmethod
    def initialize_environment(args: ArgsEmulator = None, 
                               pomdp_model=None, 
                               state_based_oracle=None, 
                               state_based_sim=None,
                               num_of_expansion_fatures = 0) -> tuple[EnvironmentWrapperVec | Environment_Wrapper, tf_py_environment.TFPyEnvironment, object]:
        if pomdp_model is None:
            pomdp_model = ExperimentInterface.initialize_prism_model(args)
        logger.info("Model initialized")
        if args.replay_buffer_option == ReplayBufferOptions.ORIGINAL_OFF_POLICY or not args.vectorized_envs_flag:
            num_envs = 1
            args.num_environments = 1
        else:
            num_envs = args.num_environments
        if args.vectorized_envs_flag:
            environment = EnvironmentWrapperVec(
                pomdp_model, args, num_envs=num_envs)
        else:
            environment = Environment_Wrapper(pomdp_model, args)
        # self.environment = Environment_Wrapper_Vec(self.pomdp_model, self.args, num_envs=num_envs)
        tf_environment = tf_py_environment.TFPyEnvironment(
            environment, check_dims=True)
        # self.tf_environment_orig = tf_py_environment.TFPyEnvironment(self.environment_orig)
        logger.info("Environment initialized")
        return environment, tf_environment, pomdp_model

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
    
    def init_vectorized_evaluation_driver_w_buffer(self, tf_environment: tf_py_environment.TFPyEnvironment, 
                                                   environment: EnvironmentWrapperVec, 
                                                   custom_policy : TFPolicy=None, num_steps=400) -> tuple[DynamicStepDriver, TrajectoryBuffer]:
        """Initialize the vectorized evaluation driver for the agent. Used for evaluation of the agent.

        Args:
            tf_environment: The TensorFlow environment object, used for simulation information.
            environment: The vectorized environment object, used for simulation information.
            num_steps: The number of steps for evaluation.
        """
        trajectory_buffer = TrajectoryBuffer(environment)
        eager = PyTFEagerPolicy(
            policy=custom_policy, use_tf_function=True, batch_time_steps=False)
        vec_driver = DynamicStepDriver(
            tf_environment,
            eager,
            observers=[trajectory_buffer.add_batched_step],
            num_steps=(1 + num_steps) * self.args.num_environments
        )
        return vec_driver, trajectory_buffer
    
    def evaluate_extracted_fsc(self, external_evaluation_result : EvaluationResults, model : str = ""):
        """Evaluates the extracted FSC. The result is saved to the self.agent.evaluation_result object."""
        from interpreters.fsc_based_interpreter import NaiveFSCPolicyExtraction
        if external_evaluation_result is None or True:
            evaluation_result = EvaluationResults()
        else:
            evaluation_result = external_evaluation_result
        extracted_fsc_policy = NaiveFSCPolicyExtraction(self.agent.wrapper, self.environment, self.tf_environment, self.args, model = model)
        if not hasattr(self, "vec_driver"):
            driver, buffer = self.init_vectorized_evaluation_driver_w_buffer(
                self.tf_environment, self.environment, custom_policy=extracted_fsc_policy, num_steps=self.args.max_steps)
        self.tf_environment.reset()
        driver.run()
        buffer.final_update_of_results(
            evaluation_result.update)
        log_evaluation_info(evaluation_result)
        external_evaluation_result.last_from_interpretation = True
        external_evaluation_result.extracted_fsc_episode_return = evaluation_result.returns_episodic[-1]
        external_evaluation_result.extracted_fsc_return = evaluation_result.returns[-1]
        external_evaluation_result.extracted_fsc_reach_prob = evaluation_result.reach_probs[-1]
        external_evaluation_result.extracted_fsc_num_episodes = evaluation_result.num_episodes[-1]
        external_evaluation_result.extracted_fsc_variance = evaluation_result.each_episode_variance[-1]
        external_evaluation_result.extracted_fsc_virtual_variance = evaluation_result.each_episode_virtual_variance[-1]
        external_evaluation_result.extracted_fsc_combined_variance = evaluation_result.combined_variance[-1]
        
        buffer.clear()

    def perform_experiment(self, with_refusing=False, model : str = "", state_based_agent : FatherAgent = None, state_based_sim : StormVecEnv= None):
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
        self.environment, self.tf_environment, stormpy_model = self.initialize_environment(
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
        # if self.args.use_rnn_less:
        #     evaluate_extracted_fsc(self.agent.evaluation_result, model = model, agent = self.agent)
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
