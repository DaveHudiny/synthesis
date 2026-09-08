from unittest import result

import stormpy

from compact_rl.robust_rl.robust_rl_tools import load_sketch

import os

import numpy as np

# RL implementation imports
from compact_rl.rl.environment.environment_wrapper_vec import EnvironmentWrapperVec
from compact_rl.rl.environment.tf_py_environment import TFPyEnvironment
from compact_rl.rl.agents.recurrent_ppo_agent import Recurrent_PPO_agent
from compact_rl.rl.agents.father_agent import FatherAgent
from compact_rl.rl.interpreters.extracted_fsc.table_based_policy import TableBasedPolicy
from compact_rl.rl.tools.args_emulator import ArgsEmulator
from compact_rl.rl.tools.evaluators import evaluate_policy_in_model
from compact_rl.rl.tests.general_test_tools import init_args
from compact_rl.robust_rl.permissive_dtmc_extractor import Permissive_DTMC_Extractor


# PAYNT implementation imports
from paynt.parser.sketch import Sketch
from paynt.rl_extension.self_interpretable_interface.black_box_extraction import BlackBoxExtractor
from paynt.rl_extension.family_extractors.direct_fsc_construction import ConstructorFSC
from paynt.quotient.fsc import FscFactored



def load_sketch(project_path):
    project_path = os.path.abspath(project_path)
    if os.path.exists(os.path.join(project_path, "sketch.templ")):
        sketch_path = os.path.join(project_path, "sketch.templ")
        properties_path = os.path.join(project_path, "sketch.props")
    else:
        if os.path.exists(os.path.join(project_path, "model-random-enabled.drn")):
            sketch_path = os.path.join(project_path, "model-random-enabled.drn")
            properties_path = os.path.join(project_path, "discounted.props")
        else:
            raise ValueError(f"Could not find sketch.templ or model-random-enabled.drn in {project_path}")
    pomdp_sketch = Sketch.load_sketch(
        sketch_path, properties_path)
    return pomdp_sketch


def create_json_file_name(project_path, seed=""):
    """
    Creates a JSON file name based on the project path.
    """
    json_path = os.path.join(project_path, f"benchmark_stats_{seed}.json")
    if os.path.exists(json_path):
        index = 0
        while os.path.exists(os.path.join(project_path, f"benchmark_stats_{seed}_{index}.json")):
            index += 1
        json_path = os.path.join(
            project_path, f"benchmark_stats_{seed}_{index}.json")
    return json_path


def init_extractor(model, args: ArgsEmulator, latent_dim=9, autlearn_extraction=True, steps_to_take=4000, training_epochs=1001) -> BlackBoxExtractor:
    """Function that initializes the FSC extractor/synthesizer.
    Args:
        args (ArgsEmulator): Arguments object containing various settings for the RL and extraction process.
        latent_dim (int, optional): Dimension of the latent space, which defines the maximum size of the FSC provided by SIG. Defaults to 9.
        autlearn_extraction (bool, optional): Selection between SIG extraction and the AALpy Alergia. Defaults to True (Alergia).
        steps_to_take (int, optional): Number of steps, that is taken in each of the parallel simulators. Defaults to 4000.
        training_epochs (int, optional): SIG training epochs irrelevant to Alergia. Defaults to 20001.

    Returns:
        BlackBoxExtractor: Initialized object that performs the SIG or Alergia extraction.
    """
    # family_quotient_numpy = FamilyQuotientNumpy(model)
    direct_extractor = BlackBoxExtractor(memory_len=latent_dim, is_one_hot=True,
                                          use_residual_connection=True, training_epochs=training_epochs,
                                          num_data_steps=steps_to_take, get_best_policy_flag=False,
                                          max_episode_len=args.max_steps,
                                          family_quotient_numpy=None,
                                          autlearn_extraction=autlearn_extraction,
                                          use_gumbel_softmax=True,
                                          non_deterministic=False)
    return direct_extractor


def fsc_extraction(model, agent: FatherAgent) -> tuple[FscFactored, TableBasedPolicy]:

    direct_extractor = init_extractor(model, agent.args, autlearn_extraction=True)
    policy = agent.get_policy(False, True)
    policy.set_greedy(True) # Ensures, that the agent selects argmax actions
    policy.set_policy_masker() # Ensures, that the agent respects the action masking during extraction
    extracted_fsc, extraction_stats = direct_extractor.clone_and_generate_fsc_from_policy( # Extraction stats is a structure that contains various statistics about the extraction process
        policy, agent.environment, agent.tf_environment)
    evaluate_policy_in_model(extracted_fsc, agent.args, agent.environment, agent.tf_environment) # We can evaluate the extracted FSC in the model to see its performance

    paynt_fsc = ConstructorFSC.construct_fsc_from_table_based_policy(extracted_fsc, pomdp_quotient=model, family_quotient_numpy=None, cut_probs=1.0) # Generate deterministic PAYNT FSC representation. 
    tf_fsc_policy = TableBasedPolicy(original_policy=policy, # TODO: Replace the original policy from the policy initialization process. 
                                     action_function=np.array(paynt_fsc.action_function), # If you provide the action and update functions (shape [memory_size, nr_observations]), you obtain standard TF Policy that can be evaluated in TF-Agents environment.
                                     update_function=np.array(paynt_fsc.update_function), 
                                     action_keywords=paynt_fsc.action_labels
                                )
    evaluate_policy_in_model(tf_fsc_policy, agent.args, agent.environment, agent.tf_environment)

    policy.set_greedy(False) # Reset the policy to non-greedy mode
    policy.set_identity_masker() # Unset the policy masker to allow all actions again, TODO: Check if this is necessary
    return paynt_fsc, tf_fsc_policy


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_path = script_dir + "/models/models_single_pomdp/network-3-8-20"
    prism_path = os.path.join(project_path, "sketch.templ")
    properties_path = os.path.join(project_path, "sketch.props")
    args = init_args(prism_path=prism_path, properties_path=properties_path,
                     use_rnn_less=False, # Use RNN-less agent (if True, the policy should be completely memoryless)
                     max_steps=601, # Max steps per episode
                     seed=None, # Random seed, for the reproducibility, set it to some integer value
                     prefer_stochastic=True, # Whether to prefer stochastic or deterministic actions during the evaluation
                    )
    # Replace by your sketch loader.
    sketch = load_sketch(project_path=project_path)

    # ---------------------------------------------------------
    # This is the learning
    model = sketch.pomdp # If you don't have POMDP, you can switch to quotient mdp or some other MDP/POMDP representations.
    # model = sketch.quotient_mdp
    # print(model)
    # exit(0)
    # print(f"DTMC from permissive policy has value {result.at(initial_state)} for formula {formula}.")
    # exit(0)


    environment = EnvironmentWrapperVec(
        model, args, num_envs=args.num_environments, enforce_compilation=True)

    tf_env = TFPyEnvironment(environment)
    agent = Recurrent_PPO_agent(
        environment=environment, tf_environment=tf_env, args=args, load=False, agent_folder="trained_agents")
    agent.train_agent(iterations=500)
    # agent.save_agent() # Ensures that the agent is saved for the later use.
    
    policy = agent.get_policy(False, True)

    
    policy.set_policy_masker() # Disallow agent to play illegal actions during the evaluation.
    # policy.set_identity_masker() # Agent can play illegal actions during the execution.

    #---------------------------------------------------------
    # This is the evaluation of the learned policy in the model.
    evaluate_policy_in_model(policy, args, environment, tf_env)
    # ---------------------------------------------------------


    #---------------------------------------------------------
    # Construction and model-checking of DTMC with permissive memoryless policy over original POMDP.
    extractor = Permissive_DTMC_Extractor(model, policy, environment)
    dtmc = extractor.extract_dtmc(num_steps=1001, nr_environments=256, threshold=0.1)
    formula = 'R{"dropped_packets"}min=?[F "goal" ]' # Example property formula for network model
    
    properties = stormpy.parse_properties(formula)
    result = stormpy.model_checking(dtmc, properties[0])
    initial_state = model.initial_states[0]
    print(f"DTMC from permissive policy has value {result.at(initial_state)} for formula {formula}.")
    #---------------------------------------------------------
    
    # This performs the extraction.

    if hasattr(sketch, 'pomdp'): # Ensure, that the sketch has a POMDP representation
        paynt_fsc, tf_fsc = fsc_extraction(sketch, agent)

    # Save the results. Now the results are stored in the same folder as the processed models, but you can change it as needed.
    json_path = create_json_file_name(project_path, seed=args.seed)
    agent.evaluation_result.save_to_json(json_path, new_pomdp=False)


if __name__ == "__main__":
    main()
