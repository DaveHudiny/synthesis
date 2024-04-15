from abc import ABC, abstractmethod

class Interpret(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_dictionary(self):
        raise NotImplementedError
