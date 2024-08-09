# File: rl_main.py
# Description: Main for Reinforcement Learning Approach. If you want to train agents,
#              you can do it here, or you can use --reinforcement-learning option within PAYNT
# Author: David Hudak
# Login: xhudak03

import pickle
from agents.father_agent import FatherAgent
from agents.random_agent import RandomTFPAgent
from agents.policies.fsc_policy import FSC_Policy, FSC
from interpreters.tracing_interpret import TracingInterpret
from interpreters.model_free_interpret import ModelFreeInterpret, ModelInfo

from agents.recurrent_ppo_agent import Recurrent_PPO_agent
from agents.recurrent_ddqn_agent import Recurrent_DDQN_agent
from agents.recurrent_dqn_agent import Recurrent_DQN_agent
from agents.ppo_with_qvalues_fsc import PPO_with_QValues_FSC

import paynt.parser.sketch
import paynt.synthesizer.synthesizer_pomdp

from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tools.evaluators import *
from environment.environment_wrapper import *
from environment.pomdp_builder import *
import tensorflow as tf
import sys
import os
import rl_parser

import logging

logger = logging.getLogger(__name__)

logging.getLogger('tensorflow').setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

sys.path.append("../")


tf.autograph.set_verbosity(0)


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class PAYNT_Playground:
    @staticmethod
    def fill_nones_in_qvalues(qvalues):
        for state in range(len(qvalues)):
            for memory in range(len(qvalues[state])):
                if qvalues[state][memory] is None:
                    qvalues[state][memory] = np.mean([qvalues[state][i] for i in range(len(qvalues[state])) if qvalues[state][i] is not None])
        return qvalues

    @classmethod
    def compute_qvalues_function(cls, sketch_path, properties_path):
        if not os.path.exists(sketch_path):
            raise ValueError(f"Sketch file {sketch_path} does not exist.")
        if not hasattr(cls, "quotient") and not hasattr(cls, "synthesizer"):
            cls.quotient = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)
            k = 2 # May be unknown?
            cls.quotient.set_imperfect_memory_size(k)
            cls.synthesizer = paynt.synthesizer.synthesizer_pomdp.SynthesizerPomdp(cls.quotient, method="ar", storm_control=None)
        assignment = cls.synthesizer.synthesize()
        # before the quotient is modified we can use this assignment to compute Q-values
        assert assignment is not None
        qvalues = cls.quotient.compute_qvalues(assignment)
        # note Q(s,n) may be None if (s,n) exists in the unfolded POMDP but is not reachable in the induced DTMC
        memory_size = len(qvalues[0])
        assert k == memory_size
        qvalues = PAYNT_Playground.fill_nones_in_qvalues(qvalues)
        return qvalues

def save_dictionaries(name_of_experiment, model, learning_method, refusing_typ, obs_action_dict, memory_dict, labels):
    """ Save dictionaries for Paynt oracle.
    Args:
        name_of_experiment (str): Name of the experiment.
        model (str): The name of the model.
        learning_method (str): The learning method.
        refusing_typ (str): Whether to use refusing when interpreting.
        obs_action_dict (dict): The observation-action dictionary.
        memory_dict (dict): The memory dictionary.
        labels (dict): The labels dictionary.
    """
    if not os.path.exists(f"{name_of_experiment}/{model}_{learning_method}/{refusing_typ}"):
        os.makedirs(
            f"{name_of_experiment}/{model}_{learning_method}/{refusing_typ}")
    with open(f"{name_of_experiment}/{model}_{learning_method}/{refusing_typ}/obs_action_dict.pickle", "wb") as f:
        pickle.dump(obs_action_dict, f)
    with open(f"{name_of_experiment}/{model}_{learning_method}/{refusing_typ}/memory_dict.pickle", "wb") as f:
        pickle.dump(memory_dict, f)
    with open(f"{name_of_experiment}/{model}_{learning_method}/{refusing_typ}/labels.pickle", "wb") as f:
        pickle.dump(labels, f)


def save_statistics_to_new_json(name_of_experiment, model, learning_method, evaluation_result: EvaluationResults, args: dict = None):
    """ Save statistics to a new JSON file.
    Args:
        name_of_experiment (str): Name of the experiment.
        model (str): The name of the model.
        learning_method (str): The learning method.
        evaluation_result (EvaluationResults): The evaluation results.
        args (dict, optional): The arguments. Defaults to None.
    """
    if args is None:
        max_steps = 300
    else:
        max_steps = args.max_steps

    evaluation_result.set_experiment_settings(
        learning_algorithm=learning_method, max_steps=max_steps)
    if not os.path.exists(f"{name_of_experiment}"):
        os.mkdir(f"{name_of_experiment}")
    if os.path.exists(f"{name_of_experiment}/{model}_{learning_method}_training.json"):
        i = 1
        while os.path.exists(f"{name_of_experiment}/{model}_{learning_method}_training_{i}.json"):
            i += 1
        evaluation_result.save_to_json(
            f"{name_of_experiment}/{model}_{learning_method}_training_{i}.json")
    else:
        evaluation_result.save_to_json(
            f"{name_of_experiment}/{model}_{learning_method}_training.json")


class ArgsEmulator:
    def __init__(self, prism_model: str = None, prism_properties: str = None, constants: str = "", discount_factor: float = 0.75,
                 encoding_method: str = "Valuations", learning_rate: float = 1.6e-3, max_steps: int = 300, evaluation_episodes: int = 20,
                 batch_size: int = 32, trajectory_num_steps: int = 32, nr_runs: int = 5000, evaluation_goal: int = 300,
                 interpretation_method: str = "Tracing", learning_method: str = "DQN",
                 save_agent: bool = True, seed: int = 123456, evaluation_antigoal: int = -300, experiment_directory: str = "experiments",
                 buffer_size: int = 5000, interpretation_granularity: int = 100, load_agent: bool = False, restart_weights: int = 0, action_filtering: bool = False,
                 illegal_action_penalty: float = -3, randomizing_illegal_actions: bool = True, randomizing_penalty: float = -1, reward_shaping: bool = False,
                 reward_shaping_model: str = "evade", agent_name="test", paynt_fsc_imitation=False, paynt_fsc_json=None, fsc_policy_max_iteration=100,
                 interpretation_folder="interpretation", experiment_name="experiment", with_refusing=None, set_ppo_on_policy=False,
                 evaluate_random_policy: bool = False, prefer_stochastic: bool = False):
        """Args emulator for the RL parser. This class is used to emulate the args object from the RL parser for the RL initializer and other stuff.
        Args:

        prism_model (str): The path to the prism model file. Defaults to None -- must be set, if not used inside of Paynt.
        prism_properties (str): The path to the prism properties file. Defaults to None -- must be set, if not used inside of Paynt.
        constants (str, optional): The constants for the model. Syntax looks like: "C1=10,C2=60". See Prism template for definable constants. Defaults to "".
        discount_factor (float, optional): The discount factor for the environment. Defaults to 1.0.
        encoding_method (str, optional): The encoding method for the observations. Defaults to "Valuations". Other possible selections are "One-Hot" and "Integer".
        learning_rate (float, optional): The learning rate. Defaults to 1e-7.
        max_steps (int, optional): The maximum steps per episode. Defaults to 100.
        evaluation_episodes (int, optional): The number of evaluation episodes. Defaults to 10.
        batch_size (int, optional): The batch size. Defaults to 32.
        trajectory_num_steps (int, optional): The number of steps for each sample trajectory. Used for training the agent. Defaults to 25.
        nr_runs (int, optional): The number of runs. Defaults to 500.
        evaluation_goal (int, optional): The evaluation goal. Defaults to 10.
        interpretation_method (str, optional): The interpretation method. Defaults to "Tracing". Other possible selection is "Model-Free", 
                                               but it is not fully functional yet.
        learning_method (str, optional): The learning method. Choices are ["DQN", "DDQN", "PPO"]. Defaults to "DQN".
        save_agent (bool, optional): Save agent model during training. Defaults to False.
        load_agent (bool, optional): Load agent model during training. Defaults to False.
        seed (int, optional): Seed for reproducibility. Defaults to 123456.
        evaluation_antigoal (int, optional): The evaluation antigoal. Defaults to -10.
        experiment_directory (str, optional): Directory for files from experiments. Defaults to "experiments".
        buffer_size (int, optional): Buffer size for the replay buffer. Defaults to 1000.
        interpretation_granularity (int, optional): The number of episodes for interpretation. Defaults to 50.
        restart_weights (int, optional): The number of restarts of weights before starting learning. Defaults to 0.
        action_filtering (bool, optional): Filtering of actions performed by the environment. 
                                           If set, the environment will filter the actions based on the current state and return negative reward. 
                                           Defaults to False.
        illegal_action_penalty (float, optional): Penalty for illegal actions. Defaults to -3.
        randomizing_illegal_actions (bool, optional): Randomize illegal actions. Defaults to True.
        randomizing_penalty (float, optional): Penalty for randomizing illegal actions. Defaults to -0.1.
        reward_shaping (bool, optional): Reward shaping. Defaults to False.
        reward_shaping_model (str, optional): Reward shaping model. Defaults to "evade". Other possible selection is "refuel".
        agent_name (str, optional): The name of the agent. Defaults to "test".
        paynt_fsc_imitation (bool, optional): Use extracted FSC from Paynt for improving data collection and imitation learning. Defaults to False.
        paynt_fsc_json (str, optional): JSON file with extracted FSC from Paynt. Defaults to None.
        fsc_policy_max_iteration (int, optional): If --paynt-fsc-imitation is selected, this parameter defines the maximum number of iterations for FSC policy training. Defaults to 100.
        interpretation_folder (str, optional): The folder for interpretation. Defaults to "interpretation".
        experiment_name (str, optional): The name of the experiment. Defaults to "experiment".
        with_refusing (bool, optional): Whether to use refusing when interpreting. Defaults to None.
        set_ppo_on_policy (bool, optional): Set PPO to on-policy. With other methods, this parameter has no effect. Defaults to False.
        prefer_stochastic (bool, optional): Prefer stochastic actions (in case of PPO) for evaluation. Defaults to False.
        """
        self.prism_model = prism_model
        self.prism_properties = prism_properties
        self.constants = constants
        self.discount_factor = discount_factor
        self.encoding_method = encoding_method
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.evaluation_episodes = evaluation_episodes
        self.interpretation_granularity = interpretation_granularity
        self.batch_size = batch_size
        self.num_steps = trajectory_num_steps
        self.nr_runs = nr_runs
        self.evaluation_goal = evaluation_goal
        self.interpretation_method = interpretation_method
        self.learning_method = learning_method
        self.save_agent = save_agent
        self.load_agent = load_agent
        self.seed = seed
        self.evaluation_antigoal = evaluation_antigoal
        self.experiment_directory = experiment_directory
        self.buffer_size = buffer_size
        self.restart_weights = restart_weights
        self.action_filtering = action_filtering
        self.illegal_action_penalty = illegal_action_penalty
        self.randomizing_illegal_actions = randomizing_illegal_actions
        self.randomizing_penalty = randomizing_penalty
        self.reward_shaping = reward_shaping
        self.reward_shaping_model = reward_shaping_model
        self.agent_name = agent_name
        self.paynt_fsc_imitation = paynt_fsc_imitation
        self.paynt_fsc_json = paynt_fsc_json
        self.fsc_policy_max_iteration = fsc_policy_max_iteration
        self.interpretation_folder = interpretation_folder
        self.experiment_name = experiment_name
        self.with_refusing = with_refusing
        self.set_ppo_on_policy = set_ppo_on_policy
        self.evaluate_random_policy = evaluate_random_policy
        self.prefer_stochastic = prefer_stochastic


class Initializer:
    def __init__(self, args: ArgsEmulator = None, pomdp_model=None, agent=None):
        if args is None:
            self.parser = rl_parser.Parser()
            self.args = self.parser.args
        else:
            self.args = args
        self.pomdp_model = pomdp_model
        self.agent = agent

    def asserts(self):
        if self.args.prism_model and not self.args.prism_properties:
            raise ValueError("Prism model is set but Prism properties are not")
        if self.args.paynt_fsc_imitation and not self.args.paynt_fsc_json:
            raise ValueError(
                "Paynt imitation is set but there is not selected any JSON FSC file.")

    def parse_properties(self):
        with open(self.args.prism_properties, "r") as f:
            lines = f.readlines()
        properties = []
        for line in lines:
            if line.startswith("//"):
                continue
            properties.append(line.strip())
        return properties

    def initialize_prism_model(self):
        properties = self.parse_properties()
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

    def initialize_environment(self, parallelized: bool = False, num_parallel_environments: int = 4):
        self.pomdp_model = self.initialize_prism_model()
        logger.info("Model initialized")
        self.environment = Environment_Wrapper(self.pomdp_model, self.args)
        if parallelized:
            tf_environment = parallel_py_environment.ParallelPyEnvironment(
                [self.environment.create_new_environment] * num_parallel_environments)
        else:
            tf_environment = tf_py_environment.TFPyEnvironment(
                self.environment)
        logger.info("Environment initialized")
        return tf_environment

    def select_best_starting_weights(self, agent: FatherAgent):
        logger.info("Selecting best starting weights")
        best_cumulative_return, best_average_last_episode_return, _ = compute_average_return(
            agent.get_evaluation_policy(), self.tf_environment, self.args.evaluation_episodes)
        agent.save_agent()
        for i in range(self.args.restart_weights):
            logger.info(f"Restarting weights {i + 1}")
            agent.reset_weights()
            cumulative_return, average_last_episode_return, _ = compute_average_return(
                agent.get_evaluation_policy(), self.tf_environment, self.args.evaluation_episodes)
            if average_last_episode_return > best_average_last_episode_return:
                best_cumulative_return = cumulative_return
                best_average_last_episode_return = average_last_episode_return
                agent.save_agent()
            elif average_last_episode_return == best_average_last_episode_return:
                if cumulative_return > best_cumulative_return:
                    best_cumulative_return = cumulative_return
                    agent.save_agent()
        logger.info(f"Best cumulative return: {best_cumulative_return}")
        logger.info(
            f"Best average last episode return: {best_average_last_episode_return}")
        logger.info("Agent with best ")
        agent.load_agent()
        return agent

    def select_agent_type(self, learning_method=None, qvalues_table=None) -> FatherAgent:
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
                self.environment, self.tf_environment, self.args, load=self.args.load_agent, agent_folder=agent_folder)
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
        elif learning_method == "PPO_FSC_Critic":
            if qvalues_table is None: # If Q-values table is not provided, compute it from the sketch and properties
                sketch_path = self.args.prism_model
                props_path = self.args.prism_properties
                qvalues_table = PAYNT_Playground.compute_qvalues_function(sketch_path, props_path)
            agent = PPO_with_QValues_FSC(
                self.environment, self.tf_environment, self.args, load=self.args.load_agent, agent_folder=agent_folder,
                qvalues_table=qvalues_table)
        else:
            raise ValueError(
                "Learning method not recognized or implemented yet.")
        return agent

    def initialize_agent(self, qvalues_table = None) -> FatherAgent:
        """Initializes the agent. The agent is initialized based on the learning method and encoding method. The agent is saved to the self.agent variable.
        It is important to have previously initialized self.environment, self.tf_environment and self.args.

        returns:
            FatherAgent: The initialized agent.
        """
        agent = self.select_agent_type(qvalues_table=qvalues_table)
        if self.args.restart_weights > 0:
            agent = self.select_best_starting_weights(agent)
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
        agent = RandomTFPAgent(
            self.environment, self.tf_environment, self.args, load=False)
        interpret = TracingInterpret(self.environment, self.tf_environment,
                                     self.args.encoding_method, self.environment._possible_observations)
        results = {}
        for refusing in [True, False]:
            result = interpret.get_dictionary(agent, refusing)
            if refusing:
                results["best_with_refusing"] = result
                results["last_with_refusing"] = result
            else:
                results["best_without_refusing"] = result
                results["last_without_refusing"] = result
        return results

    def tracing_interpretation(self, with_refusing=None):
        interpret = TracingInterpret(self.environment, self.tf_environment,
                                     self.args.encoding_method, self.environment._possible_observations)
        for quality in ["last", "best"]:
            logger.info(f"Interpreting agent with {quality} quality")
            if with_refusing == None:
                self.agent.load_agent(quality == "best")
                if self.args.learning_method == "PPO" and self.args.prefer_stochastic:
                    self.agent.set_agent_stochastic()
                elif self.args.learning_method == "PPO":
                    self.agent.set_agent_greedy()
                result[f"{quality}_with_refusing"] = interpret.get_dictionary(
                    self.agent, with_refusing=True)
                result[f"{quality}_without_refusing"] = interpret.get_dictionary(
                    self.agent, with_refusing=False)
            else:
                result = interpret.get_dictionary(self.agent, with_refusing)
        return result

    def main(self, with_refusing=False):
        try:
            self.asserts()
        except ValueError as e:
            logger.error(e)
            return
        self.tf_environment = self.initialize_environment()
        if self.args.evaluate_random_policy:  # Evaluate random policy
            return self.evaluate_random_policy()

        self.agent = self.initialize_agent()
        if self.args.learning_method == "PPO" and self.args.set_ppo_on_policy:
            self.agent.train_agent_on_policy(self.args.nr_runs)
        else:
            self.agent.train_agent_off_policy(self.args.nr_runs)
        self.agent.save_agent()
        result = {}
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initializer = Initializer()
    args = initializer.parser.args
    result = initializer.main(args.with_refusing)
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
