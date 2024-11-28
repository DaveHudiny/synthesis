# Class for abstract agent, probably redundant.
# Author: David Hudák
# Login: xhudak03
# Project: diploma-thesis
# File: abstract_agent.py

from abc import ABC, abstractmethod
from tools.args_emulator import ReplayBufferOptions


class AbstractAgent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def policy(self, time_step, policy_state=None):
        raise NotImplementedError

    @abstractmethod
    def train_agent(self, iterations: int, vectorized: bool = True, replay_buffer_option: ReplayBufferOptions = ReplayBufferOptions.ON_POLICY):
        raise NotImplementedError

    @abstractmethod
    def save_agent(self, path):
        raise NotImplementedError
