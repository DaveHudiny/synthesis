import numpy as np
from environment.pomdp_builder import *

from environment.environment_wrapper_vec import EnvironmentWrapperVec
from tools.args_emulator import ArgsEmulator
from environment.tf_py_environment import TFPyEnvironment


def init_environment(args : ArgsEmulator) -> tuple[EnvironmentWrapperVec, TFPyEnvironment]:
    prism_model = initialize_prism_model(args.prism_model, args.prism_properties, args.constants)
    env = EnvironmentWrapperVec(prism_model, args, num_envs=args.num_environments)
    tf_env = TFPyEnvironment(env)
    return env, tf_env

def init_args(prism_path, properties_path, nr_runs=101, goal_value_multiplier = 1.0) -> ArgsEmulator:
    args = ArgsEmulator(prism_model=prism_path, prism_properties=properties_path, learning_rate=1.6e-4,
                            restart_weights=0, learning_method="PPO", prefer_stochastic=False,
                            nr_runs=nr_runs, agent_name="Testus", load_agent=False,
                            evaluate_random_policy=False, max_steps=401, evaluation_goal=50, evaluation_antigoal=-20,
                            trajectory_num_steps=32, discount_factor=0.99, num_environments=256,
                            normalize_simulator_rewards=False, buffer_size=500, random_start_simulator=False,
                            batch_size=256, vectorized_envs_flag=True, perform_interpretation=True, use_rnn_less=False, model_memory_size=0,
                            stacked_observations=False)
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
