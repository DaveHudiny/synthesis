from enum import Enum
from rl_src.tools.args_emulator import ArgsEmulator
from rl_src.tools.args_emulator import ReplayBufferOptions

import logging
logger = logging.getLogger(__name__)

class RL_SAYNT_Combo_Modes(Enum):
    TRAJECTORY_MODE = 0
    QVALUES_CRITIC_MODE = 1
    QVALUES_COMBO_CRITIC_MODE = 2
    QVALUES_RANDOM_SIM_INIT_MODE = 3
    DQN_AS_QTABLE = 4
    JUMPSTART_MODE = 5
    SHAPING_MODE = 6


def init_rl_args(mode: RL_SAYNT_Combo_Modes = RL_SAYNT_Combo_Modes.QVALUES_RANDOM_SIM_INIT_MODE) -> ArgsEmulator:
    if mode == RL_SAYNT_Combo_Modes.QVALUES_CRITIC_MODE:
        # "Periodic_FSC_Neural_PPO"
        # "PPO_FSC_Critic"
        args = ArgsEmulator(load_agent=False, learning_method="PPO_FSC_Critic", encoding_method="Valuations++",
                            max_steps=400, restart_weights=0, agent_name="PAYNT", learning_rate=1e-4,
                            trajectory_num_steps=20, evaluation_goal=500, evaluation_episodes=40, evaluation_antigoal=-500,
                            discount_factor=0.99, batch_size=32)
    elif mode == RL_SAYNT_Combo_Modes.TRAJECTORY_MODE:
        args = args = ArgsEmulator(load_agent=False, learning_method="Stochastic_PPO", encoding_method="Valuations",
                            max_steps=400, restart_weights=0, agent_name="PAYNT_Trajectory", learning_rate=1.6e-4, num_environments=256,
                            trajectory_num_steps=32, evaluation_goal=50, evaluation_episodes=40, evaluation_antigoal=-20, batch_size=256,
                            discount_factor=0.99, vectorized_envs_flag=True, buffer_size=500, random_start_simulator=False, 
                            replay_buffer_option=ReplayBufferOptions.ON_POLICY)
    elif mode == RL_SAYNT_Combo_Modes.QVALUES_RANDOM_SIM_INIT_MODE:
        args = ArgsEmulator(load_agent=False, learning_method="Stochastic_PPO", encoding_method="Valuations",
                            max_steps=400, restart_weights=0, agent_name="Agent_with_random_starts", learning_rate=1e-4,
                            trajectory_num_steps=25, evaluation_goal=150, evaluation_episodes=40, evaluation_antigoal=-150,
                            discount_factor=0.99, random_start_simulator=True)
    elif mode == RL_SAYNT_Combo_Modes.DQN_AS_QTABLE:
        args = ArgsEmulator(load_agent=False, learning_method="Stochastic_PPO", encoding_method="Valuations",
                            max_steps=400, restart_weights=0, agent_name="PAYNT_BehavioralCloning", learning_rate=1.6e-4, num_environments=256,
                            trajectory_num_steps=32, evaluation_goal=50, evaluation_episodes=40, evaluation_antigoal=-20, batch_size=256,
                            discount_factor=0.99, vectorized_envs_flag=True, buffer_size=500, random_start_simulator=False, 
                            replay_buffer_option=ReplayBufferOptions.ON_POLICY)
    elif mode == RL_SAYNT_Combo_Modes.JUMPSTART_MODE:
        args = ArgsEmulator(load_agent=False, learning_method="Stochastic_PPO", encoding_method="Valuations",
                            max_steps=400, restart_weights=0, agent_name="PAYNT_JumpStart", learning_rate=1.6e-4, num_environments=256,
                            trajectory_num_steps=32, evaluation_goal=50, evaluation_episodes=40, evaluation_antigoal=-20, batch_size=256,
                            discount_factor=0.99, vectorized_envs_flag=True, buffer_size=500, random_start_simulator=False, 
                            replay_buffer_option=ReplayBufferOptions.ON_POLICY)
    elif mode == RL_SAYNT_Combo_Modes.SHAPING_MODE:
        args = ArgsEmulator(load_agent=False, learning_method="Stochastic_PPO", encoding_method="Valuations",
                            max_steps=400, restart_weights=0, agent_name="PAYNT_Shaping", learning_rate=1.6e-4, num_environments=128,
                            trajectory_num_steps=32, evaluation_goal=50, evaluation_episodes=40, evaluation_antigoal=-20, batch_size=128,
                            discount_factor=0.99, vectorized_envs_flag=True, buffer_size=500, random_start_simulator=False, 
                            replay_buffer_option=ReplayBufferOptions.ON_POLICY)
    else:
        logger.error("Mode:", mode, "not implemented yet.")
        args = ArgsEmulator(load_agent=False, learning_method="PPO", encoding_method="Valuations",
                            max_steps=400, restart_weights=0, agent_name="PAYNT", learning_rate=1e-4,
                            trajectory_num_steps=20, evaluation_goal=500, evaluation_episodes=40, evaluation_antigoal=-500,
                            discount_factor=0.99)

    return args
