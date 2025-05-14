import aalpy
import stormpy

import os
import numpy as np

from learn_aut import learn_automaton

import tensorflow as tf

from rl_src.environment.environment_wrapper_vec import EnvironmentWrapperVec
from rl_src.environment.tf_py_environment import TFPyEnvironment

from rl_src.tools.args_emulator import ArgsEmulator

from rl_src.tests.general_test_tools import initialize_prism_model

from rl_src.interpreters.extracted_fsc.table_based_policy import TableBasedPolicy
from rl_src.tools.evaluators import evaluate_policy_in_model
from rl_src.agents.recurrent_ppo_agent import Recurrent_PPO_agent

from paynt.quotient.fsc import FSC
from paynt.rl_extension.family_extractors.direct_fsc_construction import ConstructorFSC

from paynt.quotient.pomdp import PomdpQuotient
from paynt.parser.sketch import Sketch

def create_action_function(model : aalpy.MealyMachine, nr_model_states, nr_observations):
    action_function = np.zeros((nr_model_states, nr_observations), dtype=np.int32)
    for i in range(nr_model_states):
        for j in range(nr_observations):
            if f"[{j}]" in model.states[i].output_fun:
                action_function[i][j] = model.states[i].output_fun[f"[{j}]"] if model.states[i].output_fun[f"[{j}]"] != "epsilon" else 0
            else:
                action_function[i][j] = 0
    return action_function


def create_update_function(model : aalpy.MealyMachine, nr_model_states, nr_observations):
    update_function = np.zeros((nr_model_states, nr_observations), dtype=np.int32)
    state_ids = [state.state_id for state in model.states]
    state_id_to_index = {state_id: i for i, state_id in enumerate(state_ids)}
    for i in range(nr_model_states):
        for j in range(nr_observations):
            if f"[{j}]" in model.states[i].transitions:
                update_function[i][j] = state_id_to_index[model.states[i].transitions[f"[{j}]"].state_id]
            else:
                update_function[i][j] = 0
    return update_function

def create_table_based_policy(original_policy, model, nr_observations, action_labels = []) -> TableBasedPolicy:
    action_function = create_action_function(model, len(model.states), nr_observations)
    update_function = create_update_function(model, len(model.states), nr_observations)
    table_based_policy = TableBasedPolicy(
        original_policy = original_policy,
        action_function=action_function,
        update_function=update_function,
        action_keywords=action_labels,
    )
    return table_based_policy

def verify_model_by_paynt(table_fsc : TableBasedPolicy, prism_model, prism_spec):
    """
    Verifies the model by PAYNT verifier.
    """
    quotient : PomdpQuotient = Sketch.load_sketch(prism_model, prism_spec)

    fsc : FSC = ConstructorFSC.construct_fsc_from_table_based_policy(
        table_fsc,
        quotient
    )
    dtmc = quotient.get_induced_dtmc_from_fsc_vec(fsc) # If you feel that there are some issues with the precision of the DTMC, you can remove the _vec suffix
    result = stormpy.model_checking(dtmc, quotient.specification.optimality.formula)
    print(f"Result of verification: {result.at(0)}")
    

if __name__ == "__main__":
    model_name = "evade"

    model = learn_automaton(model_name)

    prism_path = f"./rl_src/models/{model_name}"
    prism_template = os.path.join(prism_path, "sketch.templ")
    prism_spec = os.path.join(prism_path, "sketch.props")

    args = ArgsEmulator(
        prism_model=prism_template,
        prism_properties=prism_spec,
        constants="",
        discount_factor=0.99,
        learning_rate=1.6e-4,
        trajectory_num_steps=32,
        num_environments=256,
        batch_size=256,
        max_steps=400
    )

    stormpy_model = initialize_prism_model(prism_template, prism_spec, constants=args.constants)
    env = EnvironmentWrapperVec(stormpy_model, args, num_envs=256)
    tf_env = TFPyEnvironment(env)
    agent = Recurrent_PPO_agent(
        env, tf_env, args
    )
    original_policy = agent.get_policy()

    nr_model_states = len(model.states)
    nr_observations = stormpy_model.nr_observations

    table_based_policy = create_table_based_policy(original_policy, model, nr_observations, env.action_keywords)
    verify_model_by_paynt(
        table_based_policy,
        prism_template,
        prism_spec,
    )

    # Evaluate the policy
    evaluation_result = evaluate_policy_in_model(
        table_based_policy,
        args,
        env,
        tf_env,
        max_steps=401,
    )


