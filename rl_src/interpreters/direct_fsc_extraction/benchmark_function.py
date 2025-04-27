

import tensorflow as tf
import tf_agents

from tests.general_test_tools import *


from interpreters.direct_fsc_extraction.direct_extractor import DirectExtractor
from interpreters.direct_fsc_extraction.cloned_fsc_actor_policy import ClonedFSCActorPolicy
from tools.specification_check import SpecificationChecker
from agents.recurrent_ppo_agent import Recurrent_PPO_agent

from tools.args_emulator import ArgsEmulator
from tools.encoding_methods import *

from interpreters.direct_fsc_extraction.data_sampler import sample_data_with_policy
from interpreters.direct_fsc_extraction.extraction_stats import ExtractionStats

from tf_agents.replay_buffers import TFUniformReplayBuffer

from tools.evaluators import evaluate_policy_in_model

from interpreters.direct_fsc_extraction.test_functions import compare_two_policies, get_encoding_functions

DEBUG = False


def run_benchmark(prism_path: str, properties_path: str, memory_size, num_data_steps=100, num_training_steps=300,
                  specification_goal="reachability", optimization_goal="max", use_one_hot=False,
                  extraction_epochs=100000, use_residual_connection=False
                  ) -> tuple[ClonedFSCActorPolicy, TFUniformReplayBuffer, ExtractionStats, Recurrent_PPO_agent]:

    args = init_args(prism_path=prism_path, properties_path=properties_path,
                     nr_runs=num_training_steps, goal_value_multiplier=1.00)
    args.agent_name = "FSC_Clone"
    args.save_agent = True
    env, tf_env = init_environment(args)
    model_name = prism_path.split("/")[-2]
    agent = Recurrent_PPO_agent(
        env, tf_env, args, agent_folder=f"{args.agent_name}/{model_name}", load=False)
    agent.train_agent(iterations=num_training_steps)
    specification_checker = SpecificationChecker(
        optimization_specification=specification_goal,
        optimization_goal=optimization_goal,
        evaluation_results=agent.evaluation_result
    )
    # agent.load_agent(True)
    # agent.evaluate_agent(vectorized = True, max_steps = 800)
    extraction_stats = ExtractionStats(
        original_policy_reachability=agent.evaluation_result.reach_probs[-1],
        original_policy_reward=agent.evaluation_result.returns[-1],
        use_one_hot=use_one_hot,
        number_of_samples=num_data_steps * args.num_environments,
        memory_size=memory_size,
        residual_connection=use_residual_connection
    )

    split_path = prism_path.split("/")
    model_name = split_path[-2]
    agent.set_agent_greedy()
    buffer = sample_data_with_policy(
        agent.wrapper, num_samples=num_data_steps,
        environment=env, tf_environment=tf_env,
    )
    cloned_actor = ClonedFSCActorPolicy(
        agent.wrapper, memory_size, agent.wrapper.observation_and_action_constraint_splitter,
        use_one_hot=use_one_hot, use_residual_connection=use_residual_connection)
    # Train the cloned actor (cloned_actor.fsc_actor) to mimic the original policy
    extraction_stats = cloned_actor.behavioral_clone_original_policy_to_fsc(
        buffer, num_epochs=extraction_epochs,
        specification_checker=specification_checker,
        environment=env, tf_environment=tf_env, args=args,
        extraction_stats=extraction_stats
    )
    fsc = DirectExtractor.extract_fsc(
        cloned_actor, env, memory_size, is_one_hot=use_one_hot)
    if DEBUG:
        buffer_test = sample_data_with_policy(
            cloned_actor, num_samples=400, environment=env, tf_environment=tf_env)
        memory_encode, memory_decode = get_encoding_functions(use_one_hot)
        compare_two_policies(cloned_actor, fsc, buffer_test, memory_encode,
                             memory_decode, memory_size, agent.environment)
    ev_res = evaluate_policy_in_model(
        fsc, args, env, tf_env, args.max_steps + 1, None)
    extraction_stats.add_fsc_result(ev_res.reach_probs[-1], ev_res.returns[-1])
    return cloned_actor, buffer, extraction_stats, agent
