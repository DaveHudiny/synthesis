class ArgsEmulator:
    def __init__(self, prism_model: str = None, prism_properties: str = None, constants: str = "", discount_factor: float = 0.75,
                 encoding_method: str = "Valuations", learning_rate: float = 1.6e-3, max_steps: int = 300, evaluation_episodes: int = 20,
                 batch_size: int = 32, trajectory_num_steps: int = 32, nr_runs: int = 5000, evaluation_goal: int = 300,
                 interpretation_method: str = "Tracing", learning_method: str = "DQN",
                 save_agent: bool = True, seed: int = 123456, evaluation_antigoal: int = -300, experiment_directory: str = "experiments",
                 buffer_size: int = 5000, interpretation_granularity: int = 100, load_agent: bool = False, restart_weights: int = 0, action_filtering: bool = False,
                 illegal_action_penalty: float = -3, randomizing_illegal_actions: bool = True, randomizing_penalty: float = -1, reward_shaping: bool = False,
                 reward_shaping_model: str = "evade", agent_name="test", paynt_fsc_imitation=False, paynt_fsc_json=None, fsc_policy_max_iteration=100,
                 interpretation_folder="interpretation", experiment_name="experiment", with_refusing=None, set_ppo_on_policy: bool = False,
                 evaluate_random_policy: bool = False, prefer_stochastic: bool = False, normalize_simulator_rewards: bool = False,
                 random_start_simulator=False):
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
        normalize_simulator_rewards (bool, optional): Normalize rewards obtained from simulator (reward = reward / goal_reward)
        random_start_simulator (bool, optional): Sets initialized simulator to work with uniformly random initial states 

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
        self.normalize_simulator_rewards = normalize_simulator_rewards
        self.random_start_simulator = random_start_simulator
