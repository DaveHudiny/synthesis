from rl_src.environment.environment_wrapper_vec import Environment_Wrapper_Vec

from tf_agents.environments import tf_py_environment

from rl_src.tests.general_test_tools import *
from rl_src.tools.args_emulator import ArgsEmulator

from rl_src.agents.recurrent_ppo_agent import Recurrent_PPO_agent

def init_environment(prism_path, properties_path, constants, args : ArgsEmulator):
    prism_model = initialize_prism_model(prism_path, properties_path, constants)
    env = Environment_Wrapper_Vec(prism_model, args, num_envs=args.num_envs)
    tf_env = tf_py_environment.TFPyEnvironment(env)
    return env, tf_env

def init_args() -> ArgsEmulator:
    args = ArgsEmulator()
    return args

if __name__ == "__main__":
    prism_path = "models/network-3-8-20/sketch.templ"
    properties_path = "models/network-3-8-20/sketch.props"
    constants = ""
    args = init_args()
    env, tf_env = init_environment(prism_path, properties_path, constants, args)
    agent = Recurrent_PPO_agent(env, tf_env, args)









