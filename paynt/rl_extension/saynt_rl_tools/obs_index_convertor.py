import stormpy


class Observation_Index_Converter:
    def __init__(self, pomdp):
        self.pomdp = pomdp
        self.states_to_observation = pomdp.observations