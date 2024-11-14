import numpy as np
from rl_src.environment.pomdp_builder import *


def get_scalarized_reward(rewards, rewards_types):
    last_reward = rewards_types[-1]
    return rewards[last_reward]


def parse_properties(prism_properties: str) -> list[str]:
    with open(prism_properties, "r") as f:
        lines = f.readlines()
    properties = []
    for line in lines:
        if line.startswith("//"):
            continue
        properties.append(line.strip())
    return properties


def initialize_prism_model(prism_model: str, prism_properties, constants: dict[str, str]):
    properties = parse_properties(prism_properties)
    pomdp_args = POMDP_arguments(
        prism_model, properties, constants)
    return POMDP_builder.build_model(pomdp_args)


special_labels = np.array(["(((sched = 0) & (t = (8 - 1))) & (k = (20 - 1)))", "goal", "done", "((x = 2) & (y = 0))",
                           "((x = (10 - 1)) & (y = (10 - 1)))"])
