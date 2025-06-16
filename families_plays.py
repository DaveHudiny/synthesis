import argparse

# using Paynt for POMDP sketches

import paynt.cli
import paynt.parser.sketch
import paynt.quotient.pomdp
import paynt.quotient.pomdp_family
import paynt.quotient.fsc
import paynt.synthesizer.synthesizer_onebyone
import paynt.synthesizer.synthesizer_ar

import payntbind

import os
import random
import cProfile, pstats

import paynt.utils
import paynt.utils.timer

from tests.general_test_tools import init_environment, init_args
from environment.environment_wrapper_vec import EnvironmentWrapperVec
from environment.tf_py_environment import TFPyEnvironment

from interpreters.direct_fsc_extraction.direct_extractor import DirectExtractor

from paynt.rl_extension.family_extractors.direct_fsc_construction import ConstructorFSC



from agents.recurrent_ppo_agent import Recurrent_PPO_agent
from tools.args_emulator import ArgsEmulator

class RobustTrainer:
    def __init__(self, args : ArgsEmulator, use_one_hot_memory=False, latent_dim=2):
        self.args = args
        self.use_one_hot_memory = use_one_hot_memory
        self.model_name = args.model_name
        self.direct_extractor = self.init_extractor(latent_dim)
        
    def init_extractor(self, latent_dim):

        direct_extractor = DirectExtractor(memory_len = latent_dim, is_one_hot=self.use_one_hot_memory,
                                           use_residual_connection=True, training_epochs=50001,
                                           num_data_steps=4001, get_best_policy_flag=False, model_name=self.model_name,
                                           max_episode_len=self.args.max_steps)
        return direct_extractor
    
    def extract_fsc(self, policy, environment, quotient) -> paynt.quotient.fsc.FSC:
        tf_environment = TFPyEnvironment(environment)
        fsc, extraction_stats = self.direct_extractor.clone_and_generate_fsc_from_policy(
            policy, environment, tf_environment)
        fsc = ConstructorFSC.construct_fsc_from_table_based_policy(fsc, quotient)
        return fsc
    
    def train_on_new_pomdp(self, pomdp, agent: Recurrent_PPO_agent):
        environment = EnvironmentWrapperVec(pomdp, self.args, num_envs=256, enforce_compilation=True)
        agent.change_environment(environment)
        agent.train_agent(10)


def load_sketch(project_path : str) -> paynt.quotient.pomdp_family.PomdpFamilyQuotient:
    project_path = os.path.abspath(project_path)
    sketch_path = os.path.join(project_path, "sketch.templ")
    properties_path = os.path.join(project_path, "sketch.props")    
    pomdp_sketch = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)
    return pomdp_sketch

def assignment_to_pomdp(pomdp_sketch : paynt.quotient.pomdp_family.PomdpFamilyQuotient, assignment):
    pomdp = pomdp_sketch.build_pomdp(assignment).model
    updated = payntbind.synthesis.restoreActionsInAbsorbingStates(pomdp)
    if updated is not None: pomdp = updated
    action_labels,_ = payntbind.synthesis.extractActionLabels(pomdp);
    num_actions = len(action_labels)
    pomdp,choice_to_true_action = payntbind.synthesis.enableAllActions(pomdp)
    observation_action_to_true_action = [None]* pomdp.nr_observations
    for state in range(pomdp.nr_states):
        obs = pomdp.observations[state]
        if observation_action_to_true_action[obs] is not None:
            continue
        observation_action_to_true_action[obs] = [None] * num_actions
        choice_0 = pomdp.transition_matrix.get_row_group_start(state)
        for action in range(num_actions):
            choice = choice_0+action
            true_action = choice_to_true_action[choice]
            observation_action_to_true_action[obs][action] = true_action
    return pomdp,observation_action_to_true_action

def random_fsc(pomdp_sketch, num_nodes):
    num_obs = pomdp_sketch.num_observations
    fsc = paynt.quotient.fsc.FSC(num_nodes, num_obs)
    # action function if of type NxZ -> Distr(Act)
    for n in range(num_nodes):
        for z in range(num_obs):
            actions = pomdp_sketch.observation_to_actions[z]
            fsc.action_function[n][z] = { action:1/len(actions) for action in actions }
    # memory update function is of type NxZ -> Distr(N) and is posterior-aware
    # note: this is currently inconsistent with definitions in paynt.quotient.fsc.FSC, but let's see how this works
    for n in range(num_nodes):
        for z in range(num_obs):
            actions = pomdp_sketch.observation_to_actions[z]
            fsc.update_function[n][z] = { n_new:1/num_nodes for n_new in range(num_nodes) }
    return fsc

def parse_args():
    parser = argparse.ArgumentParser(description="Learning for Families with PAYNT.")
    parser.add_argument(
        "--project-path",
        type=str,
        help="Path to the project directory with template and properties.",
        required=True)
    args = parser.parse_args()
    return args

def generate_agent(pomdp, args) -> Recurrent_PPO_agent:
    environment = EnvironmentWrapperVec(pomdp, args, num_envs=256, enforce_compilation=True)
    tf_env = TFPyEnvironment(environment)
    agent = Recurrent_PPO_agent(environment=environment, tf_environment=tf_env, args=args)
    return agent

def main():
    args_cmd = parse_args()
    
    paynt.utils.timer.GlobalTimer.start()

    profiling = True
    if profiling:
        pr = cProfile.Profile()
        pr.enable()

    paynt.cli.setup_logger()

    project_path = args_cmd.project_path
    pomdp_sketch = load_sketch(project_path)

    prism_path = os.path.join(project_path, "sketch.templ")
    properties_path = os.path.join(project_path, "sketch.props")

    args_emulated = init_args(prism_path=prism_path, properties_path=properties_path, nr_runs=1000)
    args_emulated.model_name = project_path.split("/")[-1]

    extractor = RobustTrainer(args_emulated, use_one_hot_memory=False, latent_dim=2)

    hole_assignment = pomdp_sketch.family.pick_any()
    pomdp, observation_action_to_true_action = assignment_to_pomdp(pomdp_sketch, hole_assignment)
    
    print(pomdp)
    agent = generate_agent(pomdp, args_emulated)


    extractor.train_on_new_pomdp(pomdp, agent)
    
    hole_assignment = pomdp_sketch.family.pick_any()
    pomdp, observation_action_to_true_action = assignment_to_pomdp(pomdp_sketch, hole_assignment)

    extractor.train_on_new_pomdp(pomdp, agent)
    
    agent.set_policy_masking()
    agent.set_agent_greedy()
    fsc = extractor.extract_fsc(agent.get_policy(), pomdp, quotient=pomdp_sketch)

    # fsc = random_fsc(pomdp_sketch, 3)
    dtmc_sketch = pomdp_sketch.build_dtmc_sketch(fsc, negate_specification=True)

    synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(dtmc_sketch)
    synthesizer.run()

    if profiling:
        pr.disable()
        stats = pr.create_stats()
        pstats.Stats(stats).sort_stats("tottime").print_stats(10)

    return 



main()