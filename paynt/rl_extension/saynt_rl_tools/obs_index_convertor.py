import stormpy

import logging

from rl_src.tools.encoding_methods import create_valuations_encoding



from paynt.parser.sketch import Sketch

logger = logging.getLogger(__name__)

class Observation_Index_Converter:
    def __init__(self, pomdp):
        self.pomdp = pomdp
        self.states_to_observations = pomdp.observations
        self.index_to_valuations = self.precompute_index_to_valuations()

    def precompute_index_to_valuations(self) -> dict:
        index_to_valuations = {}
        for i in range(self.pomdp.nr_observations):
            index_to_valuations[i] = create_valuations_encoding(i, self.pomdp)
        return index_to_valuations
    
    def observation_to_index(self, observation):
        # observation is vector of floats
        list(self.index_to_valuations.values()).index(observation)
        observation_index = list(self.index_to_valuations.keys())[list(self.index_to_valuations.values()).index(observation)]
        return observation_index

if __name__ == "__main__":
    logger.info("Testing Observation_Index_Converter")
    quotient = Sketch.load_sketch("rl_src/models/mba/sketch.templ", "rl_src/models/mba/sketch.props")
    converter = Observation_Index_Converter(quotient.pomdp)
    print(converter.index_to_valuations)
    print(converter.observation_to_index([0., 0., 1., 1., 0., 0.]))