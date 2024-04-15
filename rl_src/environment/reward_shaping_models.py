from abc import ABC, abstractmethod


class Abstract_Reward_Shaping(ABC):
    """Abstract class for reward shaping"""
    def __init__(self, stormpy_model, simulator):
        self.stormpy_model = stormpy_model
        self.simulator = simulator

    @abstractmethod
    def reward_shaping(self, simulator):
        raise NotImplementedError
    
class EvadeRewardModel(Abstract_Reward_Shaping):
    """Reward shaping for the evade model"""
    def __init__(self, stormpy_model, simulator):
        super().__init__(stormpy_model, simulator)

    def reward_shaping(self):
        """Reward shaping for the evade model"""
        pass
        
class RefuelRewardModel(Abstract_Reward_Shaping):
    """Reward shaping for the refuel model"""
    def __init__(self, stormpy_model, simulator):
        super().__init__(stormpy_model, simulator)

    def reward_shaping(self):
        """Reward shaping for the refuel model"""
        pass