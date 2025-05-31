import enum


class ReplayBufferOptions(enum.IntEnum):
    """Enum for replay buffer options. Used for setting replay buffer options."""
    ON_POLICY = 1  # Performs a num_steps steps in the environment and trains the agent on the collected data. Then clears the replay buffer.
    OFF_POLICY = 2  # Performs a single step in environment and adds it to the replay buffer
    # Performs multiple steps in single environment and adds it to the replay buffer
    ORIGINAL_OFF_POLICY = 3


class ArgsEmulator:

    def __init__(self, prism_model: str = None, prism_properties: str = None, constants: str = "", discount_factor: float = 0.75,
                 encoding_method: str = "Valuations", learning_rate: float = 8.6e-4, max_steps: int = 400, evaluation_episodes: int = 20,
                 batch_size: int = 256, trajectory_num_steps: int = 32, nr_runs: int = 4001, evaluation_goal: int = 50,
                 interpretation_method: str = "Tracing", learning_method: str = "PPO",
                 save_agent: bool = True, seed: int = 123456, evaluation_antigoal: int = -20, experiment_directory: str = "experiments",
                 buffer_size: int = 500, interpretation_granularity: int = 100, load_agent: bool = False, restart_weights: int = 0,
                 agent_name="test", paynt_fsc_imitation=False, paynt_fsc_json=None, fsc_policy_max_iteration=100,
                 interpretation_folder="interpretation", experiment_name="experiment", with_refusing=None,
                 replay_buffer_option=ReplayBufferOptions.ON_POLICY,
                 evaluate_random_policy: bool = False, prefer_stochastic: bool = False, normalize_simulator_rewards: bool = True,
                 random_start_simulator=False, num_environments: int = 256, perform_interpretation: bool = False, vectorized_envs_flag: bool = True,
                 illegal_action_penalty_per_step=-0.0002, flag_illegal_action_penalty=False, use_rnn_less=False, model_memory_size=0,
                 name_of_experiment="results_of_interpretation", continuous_enlargement=False, continuous_enlargement_step=1, init_size=6,
                 state_supporting: bool = False, train_state_estimator_continuously=False, completely_greedy=False,
                 render_if_possible : bool = False, model_name = "", 
                 predicate_automata_obs : bool = False, curiosity_automata_reward : bool = False, go_explore : bool = False, 
                 stacked_observations : bool = False, masked_training : bool = False, 
                 env_see_reward : bool = False, env_see_num_steps : bool = False, env_see_last_action : bool = False):
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
            buffer_size (int, optional): Buffer size for the replay buffer. Only used in off_policy methods. Defaults to 1000.
            interpretation_granularity (int, optional): The number of episodes for interpretation. Defaults to 50.
            restart_weights (int, optional): The number of restarts of weights before starting learning. Defaults to 0.
            agent_name (str, optional): The name of the agent. Defaults to "test".
            paynt_fsc_imitation (bool, optional): Use extracted FSC from Paynt for improving data collection and imitation learning. Defaults to False.
            paynt_fsc_json (str, optional): JSON file with extracted FSC from Paynt. Defaults to None.
            fsc_policy_max_iteration (int, optional): If --paynt-fsc-imitation is selected, this parameter defines the maximum number of iterations for FSC policy training. Defaults to 100.
            interpretation_folder (str, optional): The folder for interpretation. Defaults to "interpretation".
            experiment_name (str, optional): The name of the experiment. Defaults to "experiment".
            with_refusing (bool, optional): Whether to use refusing when interpreting. Defaults to None.
            replay_buffer_option (bool, optional): Replay buffer option. Defaults to ReplayBufferOptions.ON_POLICY (other are OFF_POLICY and ORIGINAL_OFF_POLICY).
            evaluate_random_policy (bool, optional): Evaluate random policy. Defaults to False.
            prefer_stochastic (bool, optional): Prefer stochastic actions (in case of PPO) for evaluation. Defaults to False.
            normalize_simulator_rewards (bool, optional): Normalize rewards obtained from simulator (reward = reward / goal_reward)
            random_start_simulator (bool, optional): Sets initialized simulator to work with uniformly random initial states
            num_environments (int, optional): Number of environments for vectorization. Defaults to 32.
            perform_interpretation (bool, optional): Whether to perform interpretation, or provide results for training only. Defaults to False.
            vectorized_envs_flag (bool, optional): Whether to use vectorized environments. Defaults to True.
            illegal_action_penalty_per_step (float, optional): Penalty for illegal action. Defaults to -5.0.
            flag_illegal_action_penalty (bool, optional): Whether to use illegal action penalty. Defaults to False.
            use_rnn_less (bool, optional): Whether to use RNN-less actor network. Defaults to False.
            model_memory_size (int, optional): The size of the model memory. Defaults to 10.
            name_of_experiment (str, optional): The name of the experiment. Defaults to "results_of_interpretation".
            continuous_enlargement (bool, optional): Whether to use continuous enlargement. Defaults to False.
            continuous_enlargement_step (int, optional): The step for continuous enlargement. Defaults to 1.
            init_size (int, optional): The initial size. Defaults to 6.
            state_supporting (bool, optional): Whether the learning is supported via continuous fully observable state features removal. Defaults to False.
            train_state_estimator_continuously (bool, optional): Whether to train state estimator continuously with trained policy. Defaults to False.
            completely_greedy (bool, optional): Whether to use completely greedy policy. Defaults to False.
            render_if_possible (bool, optional): Whether to render the environment if possible. Defaults to False.
            model_name (str, optional): The name of the model. Defaults to "".
            predicate_automata (bool, optional): Whether to use predicate automata. Extends the observation space. Defaults to False.
            curiosity_automata_reward (bool, optional): Whether to use curiosity automata reward. Defaults to False.
            go_explore (bool, optional): Whether to use Go-Explore based on predicate automata. Defaults to False.
            stacked_observations (bool, optional): Whether to use stacked observations. Defaults to False.
            masked_training (bool, optional): Whether to use masking of illegal actions during training. Defaults to False.
            env_see_reward (bool, optional): Whether the environment provides reward in the observation space. Defaults to False.
            env_see_num_steps (bool, optional): Whether the environment provides number of steps in the observation space. Defaults to False.
            env_see_last_action (bool, optional): Whether the environment provides last action in the observation space. Defaults to False.

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
        self.trajectory_num_steps = trajectory_num_steps
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
        self.agent_name = agent_name
        self.paynt_fsc_imitation = paynt_fsc_imitation
        self.paynt_fsc_json = paynt_fsc_json
        self.fsc_policy_max_iteration = fsc_policy_max_iteration
        self.interpretation_folder = interpretation_folder
        self.experiment_name = experiment_name
        self.with_refusing = with_refusing
        self.replay_buffer_option = replay_buffer_option
        self.evaluate_random_policy = evaluate_random_policy
        self.prefer_stochastic = prefer_stochastic
        self.normalize_simulator_rewards = normalize_simulator_rewards
        self.random_start_simulator = random_start_simulator
        self.num_environments = num_environments
        self.perform_interpretation = perform_interpretation
        self.vectorized_envs_flag = vectorized_envs_flag
        self.illegal_action_penalty_per_step = illegal_action_penalty_per_step
        self.flag_illegal_action_penalty = flag_illegal_action_penalty
        self.use_rnn_less = use_rnn_less
        self.name_of_experiment = name_of_experiment
        self.continuous_enlargement = continuous_enlargement
        self.continuous_enlargement_step = continuous_enlargement_step
        self.init_size = init_size
        self.state_supporting = state_supporting
        self.train_state_estimator_continuously = train_state_estimator_continuously
        self.completely_greedy = completely_greedy
        self.render_if_possible = render_if_possible
        self.model_name = model_name
        self.predicate_automata_obs = predicate_automata_obs
        self.curiosity_automata_reward = curiosity_automata_reward
        self.go_explore = go_explore
        self.use_stacked_observations = stacked_observations
        self.masked_training = masked_training
        self.env_see_reward = env_see_reward
        self.env_see_num_steps = env_see_num_steps
        self.env_see_last_action = env_see_last_action
