# Description: This file contains the encoding methods used to encode the observations and actions for the agents.
# Author: David Hudák
# Login: xhudak03
# File: encoding_methods.py

import numpy as np
import json

class EncodingMethods:
    INTEGER = 1
    VALUATIONS = 2
    VALUATIONS_PLUS = 3
    ONE_HOT_ENCODING = 4

def observation_and_action_constraint_splitter(observation):
    return observation["observation"], observation["mask"]


def create_one_hot_encoding(observation, possible_observations):
    observation_vector = np.zeros(
        shape=(len(possible_observations),), dtype=np.float32)
    observation_vector[observation] = 1.0
    return observation_vector


def create_valuations_encoding(observation, stormpy_model):
    valuations_json = stormpy_model.observation_valuations.get_json(
        observation)
    parsed_valuations = json.loads(str(valuations_json))
    vector = []
    for key in parsed_valuations:
        if type(parsed_valuations[key]) == bool:
            if parsed_valuations[key]:
                vector.append(1.0)
            else:
                vector.append(0.0)
        else:
            vector.append(float(parsed_valuations[key]))
    return np.array(vector, dtype=np.float32)

def create_valuations_encoding_plus(observation, stormpy_model):
    valuations_json = stormpy_model.observation_valuations.get_json(
        observation)
    parsed_valuations = json.loads(str(valuations_json))
    vector = []
    vector.extend([1.0 if isinstance(val, bool) and val else 0.0 if isinstance(val, bool) else float(val) for val in parsed_valuations.values()])
    vector.append(observation / stormpy_model.nr_observations)
    return np.array(vector, dtype=np.float32)