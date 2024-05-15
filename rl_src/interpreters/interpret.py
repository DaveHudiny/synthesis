# This file contains the abstract class for the interpreters, which are used to interpret the observations and actions for the agents.
# Author: David Hudák
# Login: xhudak03
# File: interpret.py

from abc import ABC, abstractmethod

class Interpret(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_dictionary(self):
        raise NotImplementedError
