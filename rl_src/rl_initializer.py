import rl_parser

import logging
import os


from environment.pomdp_builder import *
from environment.environment_wrapper import *
from agents.evaluators import *

from tf_agents.environments import tf_py_environment
from agents.recurrent_dqn_agent import Recurrent_DQN_agent
from agents.recurrent_ddqn_agent import Recurrent_DDQN_agent
from agents.recurrent_ppo_agent import Recurrent_PPO_agent
from agents.stochastic_dqn import Stochastic_DQN
from interpreters.model_free_interpret import ModelFreeInterpret, ModelInfo
from interpreters.tracing_interpret import TracingInterpret

from agents.policies.fsc_policy import FSC_Policy, FSC

import pickle

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class ArgsEmulator:
    def __init__(self, prism_model: str, prism_properties: str, constants: str = "", discount_factor: float = 0.75,
                 encoding_method: str = "Valuations", learning_rate: float = 1.6e-5, max_steps: int = 300, evaluation_episodes: int = 20,
                 batch_size: int = 32, trajectory_num_steps: int = 16, nr_runs: int = 5000, evaluation_goal: int = 300,
                 interpretation_method: str = "Tracing", log_dir: str = "logs", log_filename: str = "log.txt", learning_method: str = "DQN",
                 save_model_drn: bool = False, save_agent: bool = True, seed: int = 123456, evaluation_antigoal: int = -300, experiment_directory: str = "experiments",
                 buffer_size: int = 10000, interpretation_granularity: int = 100, load_agent: bool = False, restart_weights: int = 0, action_filtering: bool = False,
                 illegal_action_penalty: float = -3, randomizing_illegal_actions: bool = True, randomizing_penalty: float = -1, reward_shaping: bool = False,
                 reward_shaping_model: str = "evade", agent_name="test", using_logits=False, paynt_fsc_imitation=False, paynt_fsc_json=None, fsc_policy_max_iteration=100):
        """Args emulator for the RL parser. This class is used to emulate the args object from the RL parser for the RL initializer and other stuff.
        Args:

        prism_model (str): The path to the prism model file.
        prism_properties (str): The path to the prism properties file.
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
        log_dir (str, optional): The log directory. Defaults to "logs".
        log_filename (str, optional): The log filename. Defaults to "log.txt".
        learning_method (str, optional): The learning method. Choices are ["DQN", "DDQN", "PPO"]. Defaults to "DQN".
        save_model_drn (bool, optional): Save environment model to drn file. If set, the file will be saved to the specified file. Defaults to False.
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
        using_logits (bool, optional): Using logits, compatibile with PPO. Defaults to False.
        paynt_fsc_imitation (bool, optional): Use extracted FSC from Paynt for improving data collection and imitation learning. Defaults to False.
        paynt_fsc_json (str, optional): JSON file with extracted FSC from Paynt. Defaults to None.
        fsc_policy_max_iteration (int, optional): If --paynt-fsc-imitation is selected, this parameter defines the maximum number of iterations for FSC policy training. Defaults to 100.

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
        self.log_dir = log_dir
        self.log_filename = log_filename
        self.learning_method = learning_method
        self.save_model_drn = save_model_drn
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
        self.using_logits = using_logits
        self.paynt_fsc_imitation = paynt_fsc_imitation
        self.paynt_fsc_json = paynt_fsc_json
        self.fsc_policy_max_iteration = fsc_policy_max_iteration


class Initializer:
    def __init__(self, args: ArgsEmulator = None):
        if args is None:
            self.parser = rl_parser.Parser()
            self.args = self.parser.args
        else:
            self.args = args
        self.pomdp_model = None
        self.agent = None

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
        self.pomdp_model = POMDP_builder.build_model(
            pomdp_args)

    def save_model_drn(self):
        stormpy.export_to_drn(self.pomdp_model, f"{args.save_model_drn}.drn")
        with open(f"{args.save_model_drn}.props", "w") as f:
            f.write(self.raw_formula.__str__())

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

    def initialize_environment(self):
        self.initialize_prism_model()
        logging.info("Model initialized")
        self.environment = Environment_Wrapper(self.pomdp_model, self.args)
        self.tf_environment = tf_py_environment.TFPyEnvironment(
            self.environment)
        logging.info("Environment initialized")

    def select_best_starting_weights(self):
        logging.info("Selecting best starting weights")
        best_cumulative_return, best_average_last_episode_return = compute_average_return(
            self.agent.select_evaluated_policy(), self.tf_environment, self.args.evaluation_episodes, self.args.using_logits)
        self.agent.save_agent()
        for i in range(self.args.restart_weights):
            logging.info(f"Restarting weights {i + 1}")
            self.agent.reset_weights()
            cumulative_return, average_last_episode_return = compute_average_return(
                self.agent.select_evaluated_policy(), self.tf_environment, self.args.evaluation_episodes, self.args.using_logits)
            if average_last_episode_return > best_average_last_episode_return:
                best_cumulative_return = cumulative_return
                best_average_last_episode_return = average_last_episode_return
                self.agent.save_agent()
            elif average_last_episode_return == best_average_last_episode_return:
                if cumulative_return > best_cumulative_return:
                    best_cumulative_return = cumulative_return
                    self.agent.save_agent()
        logging.info(f"Best cumulative return: {best_cumulative_return}")
        logging.info(
            f"Best average last episode return: {best_average_last_episode_return}")
        logging.info("Agent with best ")
        self.agent.load_agent()

    def select_agent_type(self, learning_method=None):
        """Selects the agent type based on the learning method and encoding method in self.args. The agent is saved to the self.agent variable.
        
        Args:
            learning_method (str, optional): The learning method. If set, the learning method is used instead of the one from the args object. Defaults to None.
        Raises:
            ValueError: If the learning method is not recognized or implemented yet."""
        if learning_method is None:
            learning_method = self.args.learning_method
        agent_folder = f"./trained_agents/{self.args.agent_name}_{self.args.learning_method}_{self.args.encoding_method}"
        if learning_method == "DQN":
            self.agent = Recurrent_DQN_agent(
                self.environment, self.tf_environment, self.args, load=self.args.load_agent, agent_folder=agent_folder)
        elif learning_method == "DDQN":
            self.agent = Recurrent_DDQN_agent(
                self.environment, self.tf_environment, self.args, load=self.args.load_agent, agent_folder=agent_folder)
        elif learning_method == "PPO":
            self.agent = Recurrent_PPO_agent(
                self.environment, self.tf_environment, self.args, load=self.args.load_agent, agent_folder=agent_folder)
        elif learning_method == "Stochastic_DQN":
            self.agent = Stochastic_DQN(
                self.environment, self.tf_environment, self.args, load=self.args.load_agent, agent_folder=agent_folder)
        else:
            raise ValueError(
                "Learning method not recognized or implemented yet.")

    def initialize_agent(self):
        """Initializes the agent. The agent is initialized based on the learning method and encoding method. The agent is saved to the self.agent variable.
        It is important to have previously initialized self.environment, self.tf_environment and self.args."""
        self.select_agent_type()
        if self.args.restart_weights > 0:
            self.select_best_starting_weights()

    def initialize_fsc_agent(self):
        with open("FSC_experimental.json", "r") as f:
            fsc_json = json.load(f)
        fsc = FSC.from_json(fsc_json)
        action_keywords = self.environment.action_keywords
        policy = FSC_Policy(self.tf_environment, fsc,
                            tf_action_keywords=action_keywords)
        return policy

    def main(self, with_refusing=False):
        # random.seed(self.args.seed)
        try:
            self.asserts()
        except ValueError as e:
            logging.error(e)
            return

        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)  
        logging.basicConfig(filename=self.args.log_dir + "/" + self.args.log_filename,
                            level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

        if self.args.save_model_drn:
            self.save_model_drn()
        self.initialize_environment()
        self.initialize_agent()
        self.agent.train_agent(self.args.nr_runs)
        self.agent.save_agent()
        result = {}
        if self.args.interpretation_method == "Tracing":
            interpret = TracingInterpret(self.environment, self.tf_environment,
                                         self.args.encoding_method, self.environment._possible_observations, self.args.using_logits)
            for quality in ["last", "best"]:
                if with_refusing == None:
                    self.agent.load_agent(quality == "best")
                    result[f"{quality}_with_refusing"] = interpret.get_dictionary(
                        self.agent, with_refusing=True)
                    result[f"{quality}_without_refusing"] = interpret.get_dictionary(
                        self.agent, with_refusing=False)
                else:
                    result = interpret.get_dictionary(self.agent, with_refusing)
            self.tf_environment.close()
        else:
            raise ValueError(
                "Interpretation method not recognized or implemented yet.")
        return result


if __name__ == "__main__":
    initializer = Initializer()
    initializer.main()
