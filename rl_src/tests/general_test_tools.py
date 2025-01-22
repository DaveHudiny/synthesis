import numpy as np
from rl_src.environment.pomdp_builder import *

from rl_src.environment.environment_wrapper_vec import Environment_Wrapper_Vec
from rl_src.tools.args_emulator import ArgsEmulator
from rl_src.environment.tf_py_environment import TFPyEnvironment


def init_environment(args : ArgsEmulator) -> tuple[Environment_Wrapper_Vec, TFPyEnvironment]:
    prism_model = initialize_prism_model(args.prism_model, args.prism_properties, args.constants)
    env = Environment_Wrapper_Vec(prism_model, args, num_envs=args.num_environments)
    tf_env = TFPyEnvironment(env)
    return env, tf_env

def init_args(prism_path, properties_path) -> ArgsEmulator:
    args = ArgsEmulator(prism_model=prism_path, prism_properties=properties_path, num_environments=16, batch_size=4)
    return args


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
