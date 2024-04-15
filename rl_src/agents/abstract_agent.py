from abc import ABC, abstractmethod

class AbstractAgent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def policy(self, time_step, policy_state=None):
        raise NotImplementedError
    
    @abstractmethod
    def train_agent(self, num_iterations):
        raise NotImplementedError
    
    @abstractmethod
    def save_agent(self, path):
        raise NotImplementedError
    