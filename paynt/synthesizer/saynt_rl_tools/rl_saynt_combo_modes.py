from enum import Enum

class RL_SAYNT_Combo_Modes(Enum):
    TRAJECTORY_MODE = 0
    QVALUES_CRITIC_MODE = 1
    QVALUES_COMBO_CRITIC_MODE = 2
    QVALUES_RANDOM_SIM_INIT_MODE = 3