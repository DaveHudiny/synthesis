import argparse

import stormpy

import paynt.parser.sketch
import paynt.quotient.fsc

import payntbind
import os

import paynt.synthesizer.synthesizer_onebyone

from paynt.synthesizer.synthesizer_pomdp import SynthesizerPomdp
from paynt.quotient.storm_pomdp_control import StormPOMDPControl

from paynt.quotient.pomdp import PomdpQuotient
from paynt.quotient.pomdp_family import PomdpFamilyQuotient


from compact_rl.rl.agents.recurrent_ppo_agent import Recurrent_PPO_agent


import numpy as np

import logging

from paynt.parser.prism_parser import PrismParser
from paynt.verification.property import construct_property

from paynt.quotient.fsc import FscFactored

logger = logging.getLogger(__name__)


def convert_all_fsc_keys_to_int(fsc: FscFactored):
    """
    Converts all keys in the FSC to integers.
    This is necessary for compatibility with the PAYNT library.
    """
    for n in range(fsc.num_nodes):
        for z in range(fsc.num_observations):
            fsc.action_function[n][z] = {
                int(action): prob for action, prob in fsc.action_function[n][z].items()}
            fsc.update_function[n][z] = {
                int(n_new): prob for n_new, prob in fsc.update_function[n][z].items()}
    return fsc


def generate_table_based_fsc_from_paynt_fsc(paynt_fsc: FscFactored, original_action_space: str, agent: Recurrent_PPO_agent):
    from compact_rl.rl.interpreters.extracted_fsc.table_based_policy import TableBasedPolicy
    # Convert [[{}]] to [[[]]] format
    minusor = 0
    if "__no_label__" in paynt_fsc.action_labels:
        minusor = 1
    action_function = np.zeros((paynt_fsc.num_nodes, paynt_fsc.num_observations, len(
        paynt_fsc.action_labels) - minusor), dtype=np.float32)
    for n in range(paynt_fsc.num_nodes):
        for z in range(paynt_fsc.num_observations):
            for action, prob in paynt_fsc.action_function[n][z].items():
                action_label = paynt_fsc.action_labels[int(action)]
                if action_label == "__no_label__":
                    continue
                else:
                    action_function[n][z][original_action_space.index(
                        action_label)] = prob
    
    # Fix the action_function (normalize)
    for n in range(paynt_fsc.num_nodes):
        for z in range(paynt_fsc.num_observations):
            total = np.sum(action_function[n][z])
            if total > 0:
                action_function[n][z] /= total
            else:
                # If no action is defined, assign uniform distribution
                num_actions = action_function.shape[2]
                action_function[n][z] = np.ones(num_actions, dtype=np.float32) / num_actions
    
    
    update_function = np.zeros(
        (paynt_fsc.num_nodes, paynt_fsc.num_observations, paynt_fsc.num_nodes), dtype=np.float32)
    
    for n in range(paynt_fsc.num_nodes):
        for z in range(paynt_fsc.num_observations):
            for n_new, prob in paynt_fsc.update_function[n][z].items():
                update_function[n][z][int(n_new)] = prob
    # Fix the update_function (normalize)
    for n in range(paynt_fsc.num_nodes):
        for z in range(paynt_fsc.num_observations):
            total = np.sum(update_function[n][z])
            if total > 0:
                update_function[n][z] /= total
            else:
                # If no update is defined, assign uniform distribution
                num_nodes = update_function.shape[2]
                update_function[n][z] = np.ones(num_nodes, dtype=np.float32) / num_nodes
    # Create the table-based policy
    table_based_policy = TableBasedPolicy(
        original_policy=agent.get_policy(False, True),
        action_function=action_function,
        update_function=update_function,
        action_keywords=original_action_space)

    return table_based_policy


def construct_reachability_spec(sketch_path):
    prism, _ = PrismParser.load_sketch_prism(sketch_path)

    prop = construct_property(stormpy.parse_properties_for_prism_program(
        f"Pmin=? [F goal]", prism)[0], 0)

    spec = paynt.verification.property.Specification([prop])

    return spec


def load_sketch(project_path):
    project_path = os.path.abspath(project_path)
    sketch_path = os.path.join(project_path, "sketch.templ")
    properties_path = os.path.join(project_path, "sketch.props")
    pomdp_sketch = paynt.parser.sketch.Sketch.load_sketch(
        sketch_path, properties_path)
    return pomdp_sketch


def assignment_to_pomdp(pomdp_sketch, assignment):
    sub_pomdp = pomdp_sketch.build_pomdp(assignment)
    pomdp = sub_pomdp.model
    state_to_quotient_state = sub_pomdp.quotient_state_map
    updated = payntbind.synthesis.restoreActionsInAbsorbingStates(pomdp)
    if updated is not None:
        pomdp = updated
    action_labels, _ = payntbind.synthesis.extractActionLabels(pomdp)
    num_actions = len(action_labels)
    return pomdp, None, state_to_quotient_state


def random_fsc(pomdp_sketch, num_nodes):
    num_obs = pomdp_sketch.num_observations
    fsc = paynt.quotient.fsc.FscFactored(num_nodes, num_obs)
    # action function if of type NxZ -> Distr(Act)
    for n in range(num_nodes):
        for z in range(num_obs):
            actions = pomdp_sketch.observation_to_actions[z]
            fsc.action_function[n][z] = {
                action: 1/len(actions) for action in actions}
    # memory update function is of type NxZ -> Distr(N) and is posterior-aware
    # note: this is currently inconsistent with definitions in paynt.quotient.fsc.FSC, but let's see how this works
    for n in range(num_nodes):
        for z in range(num_obs):
            actions = pomdp_sketch.observation_to_actions[z]
            fsc.update_function[n][z] = {
                n_new: 1/num_nodes for n_new in range(num_nodes)}
    return fsc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Learning for Families with PAYNT.")
    parser.add_argument(
        "--project-path",
        type=str,
        help="Path to the project directory with template and properties.",
        required=True)
    parser.add_argument(
        "--batched-vec-storm",
        action="store_true",
        help="Use batched vectorized Storm environment.")
    parser.add_argument(
        "--lstm-width",
        type=int,
        default=32,
        help="Width of the LSTM layers.")
    parser.add_argument(
        "--extraction-method",
        type=str,
        choices=["alergia", "si-g", "vq-vae"],
        default="alergia",
        help="Method to use for extraction. Default is 'alergia'.")
    parser.add_argument(
        "--use-best-policy",
        action="store_true",
        help="Use the best policy found during training for extraction.")
    parser.add_argument(
        "--geometric-batched-vec-storm",
        action="store_true",
        help="Use geometric distribution for batched vectorized Storm environment.")
    parser.add_argument(
        "--without-extraction",
        action="store_true",
        help="Run the experiment without extraction.")
    parser.add_argument(
        "--periodic-restarts",
        action="store_true",
        help="Use periodic restarts during training.")
    parser.add_argument(
        "--noisy-observations",
        action="store_true",
        help="Use noisy observations during training.")
    parser.add_argument(
        "--single-pomdp-setting",
        action="store_true",
        help="Experimental setting, where only a single POMDP is selected."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default is 42."
    )
    parser.add_argument(
        "--with-gru",
        action="store_true",
        default=False,
        help="Use GRU extraction for the robust RL agent to compare with."
    )
    args = parser.parse_args()
    return args


def deterministic_fsc_to_stochastic_fsc(pomdp_sketch, fsc):
    """
    Self-explanatory, map FSC with deterministic functions to a stochastic FSC with Dirac distributions.
    """
    for n in range(fsc.num_nodes):
        for z in range(fsc.num_observations):
            if fsc.is_deterministic:
                action = fsc.action_function[n][z]
                action_label = fsc.action_labels[action]
                family_action = pomdp_sketch.action_labels.index(action_label)
                fsc.action_function[n][z] = {int(family_action): 1.0}
                fsc.update_function[n][z] = {
                    int(fsc.update_function[n][z]): 1.0}
    # assert all([len(fsc.action_function[n]) == pomdp_sketch.nO for n in range(fsc.num_nodes)])
    fsc.is_deterministic = False
    return fsc


def create_json_file_name(project_path, seed = ""):
    """
    Creates a JSON file name based on the project path.
    """
    json_path = os.path.join(project_path, f"benchmark_stats_{seed}.json")
    if os.path.exists(json_path):
        index = 0
        while os.path.exists(os.path.join(project_path, f"benchmark_stats_{seed}_{index}.json")):
            index += 1
        json_path = os.path.join(project_path, f"benchmark_stats_{seed}_{index}.json")
    return json_path


def generate_heatmap_complete(pomdp_sketch, fsc):
    hole_assignments_to_test = []
    for sub_family in pomdp_sketch.family.all_combinations():
        assignment = pomdp_sketch.family.construct_assignment(sub_family)
        pomdp, _, _ = assignment_to_pomdp(pomdp_sketch, assignment)
        hole_assignments_to_test.append(assignment)
    dtmc_sketch = pomdp_sketch.build_dtmc_sketch(
        fsc, negate_specification=True)
    one_by_one = paynt.synthesizer.synthesizer_onebyone.SynthesizerOneByOne(
        dtmc_sketch)
    evaluations = []
    for i, assignment in enumerate(hole_assignments_to_test):
        evaluations.append(one_by_one.evaluate(
            assignment, keep_value_only=True))
    return evaluations, hole_assignments_to_test

def compute_policy_by_saynt(pomdp, original_pomdp_sketch : PomdpFamilyQuotient):
    PomdpQuotient.initial_memory_size = 1
    PomdpQuotient.posterior_aware = False
    pomdp_quotient = PomdpQuotient(pomdp, original_pomdp_sketch.specification)

    storm_control = StormPOMDPControl()
    storm_options = "cutoff"
    iterative_storm = [900, 8, 3]
    use_storm_cutoffs = False
    get_storm_result = None
    unfold_strategy_storm = "storm"
    prune_storm = False

    storm_control.set_options(
        storm_options, get_storm_result, iterative_storm, use_storm_cutoffs,
        unfold_strategy_storm, prune_storm
    )
    method = "ar"

    synthesizer = SynthesizerPomdp(pomdp_quotient, method, storm_control)
    synthesizer.run()

