import stormpy.storage as Storage

from paynt.rl_extension.saynt_controller.saynt_modes import SAYNT_Modes
from tf_agents.trajectories import StepType

import logging

logger = logging.getLogger(__name__)



class SAYNT_Step:
    """Class for step in SAYNT algorithm.
    """

    def __init__(self, action=0, observation=0, state: Storage.SparseModelState = None,
                 new_mode: SAYNT_Modes = SAYNT_Modes.BELIEF, tf_step_type: StepType = StepType.FIRST,
                 reward: float = 1, fsc_memory=0, integer_observation: int = 0):
        """Initialization of the step.
        Args:
            action: Action.
            observation: Observation.
            reward: Reward.
            state: Storm state.
        """
        self.action = action
        self.observation = observation
        self.state = state
        self.new_mode = new_mode
        self.tf_step_type = tf_step_type
        self.reward = reward
        self.fsc_memory = fsc_memory
        self.integer_observation = integer_observation

    def __str__(self):
        strc1 = f"Action: {self.action}, Observation: {self.observation}, state: {self.state}, new mode: {self.new_mode}, tf_step_type:, {self.tf_step_type}, "
        strc2 = f"reward {self.reward}, fsc_memory: {self.fsc_memory}"
        return strc1 + strc2