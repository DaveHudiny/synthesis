import sys
sys.path.append("../")

from environment.environment_wrapper_vec import EnvironmentWrapperVec

from tf_agents.environments import tf_py_environment

from tests.general_test_tools import *
from tools.args_emulator import ArgsEmulator, ReplayBufferOptions

from agents.recurrent_ppo_agent import Recurrent_PPO_agent

def init_environment(args : ArgsEmulator):
    prism_model = initialize_prism_model(args.prism_model, args.prism_properties, args.constants)
    env = EnvironmentWrapperVec(prism_model, args, num_envs=args.num_environments)
    tf_env = tf_py_environment.TFPyEnvironment(env)
    return env, tf_env

def init_args(prism_path, properties_path) -> ArgsEmulator:
    args = ArgsEmulator(prism_model=prism_path, prism_properties=properties_path, num_environments=8, batch_size=4)
    return args

def perform_tests():
    prism_path = "./models/network-3-8-20/sketch.templ"
    properties_path = "./models/network-3-8-20/sketch.props"
    args = init_args(prism_path=prism_path, properties_path=properties_path)
    env, tf_env = init_environment(args)
    agent = Recurrent_PPO_agent(env, tf_env, args)
    agent.train_agent(101, vectorized=True, replay_buffer_option=args.replay_buffer_option)

    









