
import vec_storm.simulator
from environment.pomdp_builder import *
from stormpy import simulator
import vec_storm

from tests.general_test_tools import get_scalarized_reward, special_labels, initialize_prism_model


import numpy as np
import random

import json



def create_valuations_encoding(observation, stormpy_model):
    valuations_json = stormpy_model.observation_valuations.get_json(
        observation)
    parsed_valuations = json.loads(str(valuations_json))
    vector = []
    for key in parsed_valuations:
        if type(parsed_valuations[key]) == bool:
            if parsed_valuations[key]:
                vector.append(1.0)
            else:
                vector.append(0.0)
        else:
            vector.append(float(parsed_valuations[key]))
    return np.array(vector, dtype=np.float32)





# Compare Vec_Storm environment and storm environment with the same model and same random policy

def test_number_of_available_actions_init(vec_available, storm_available_range):
    vec_available = sum(sum(vec_available))
    storm_available = 0
    for _ in storm_available_range:
        storm_available += 1
    assert vec_available == storm_available, f"Number of available actions is different: {vec_available} vs {storm_available}"


def generate_uniform_action_given_range(action_range):
    action = np.random.choice(action_range)
    return action

def generate_uniform_action_given_mask(mask):
    number_of_available_actions = sum(sum(mask))
    action = random.choice(range(number_of_available_actions))
    index = -1
    for i in range(len(mask[0])):
        if mask[0][i]:
            index += 1
        if index == action:
            return np.int32(i)
        
    return -1

def intersection_with_special_labels(labels):
    return [label for label in labels if label in special_labels]

def compare_overall_performance(vec_storm_simulator, storm_simulator):
    num_episodes = 500
    max_steps = 400
    _, allowed_actions, _ = vec_storm_simulator.reset()
    storm_num_goals = 0
    vec_num_goals = 0
    rewards_cumulative_storm = 0.0
    rewards_cumulative_vec = 0.0
    for _ in range(num_episodes):
        _ = storm_simulator.restart()
        _ = vec_storm_simulator.reset()
        storm_rand_action = generate_uniform_action_given_range(storm_simulator.available_actions())
        vec_rand_action = generate_uniform_action_given_mask(allowed_actions)
        storm_running = True
        vec_running = True
        

        for _ in range(max_steps):
            if storm_running:
                storm_observation, rewards, labels = storm_simulator.step(int(storm_rand_action))
                storm_rand_action = generate_uniform_action_given_range(storm_simulator.available_actions())
                rewards_cumulative_storm += rewards[0]

            if vec_running:
                actions = np.array([vec_rand_action])
                observations, rewards, done, truncated, allowed_actions, metalabels = vec_storm_simulator.step(actions=actions)
                vec_rand_action = generate_uniform_action_given_mask(allowed_actions)
                rewards_cumulative_vec += rewards[0]
                if done[0]:
                    print(rewards)

            
            intersection = intersection_with_special_labels(labels)
            if storm_running and len(intersection) > 0:
                storm_num_goals += 1
                storm_running = False
            if vec_running and metalabels[0]:
                vec_num_goals += 1
                vec_running = False

            if storm_simulator.is_done():
                storm_running = False
            if done[0]:
                vec_running = False

            if not storm_running and not vec_running:
                break

    average_storm_goals = storm_num_goals / num_episodes
    average_vec_goals = vec_num_goals / num_episodes
    if not np.isclose(average_storm_goals, average_vec_goals, atol=0.1):
        print(f"Average number of goals is different")
    else:
        print("Average number of goals is similar")
    print("Average number of storm goals", average_storm_goals)
    print("Average number of vectorized goals", average_vec_goals)

    rewards_cumulative_storm /= num_episodes
    rewards_cumulative_vec /= num_episodes
    
    # assert np.isclose(rewards_cumulative_storm, rewards_cumulative_vec, rtol=0.2), f"Average cumulative rewards are different, storm: {rewards_cumulative_storm} vs vec_storm: {rewards_cumulative_vec}"
    if not np.isclose(rewards_cumulative_storm, rewards_cumulative_vec, rtol=0.2):
        print(f"Average cumulative rewards are different")
    else:
        print("Average cumulative rewards are similar")
    print("Average cumulative rewards storm", rewards_cumulative_storm)
    print("Average cumulative rewards vec_storm", rewards_cumulative_vec)


def test_reality_of_rewards_single(vec_storm_simulator, reward_model):  
    max_steps = 5000
    print("Testing reality of rewards for model " + reward_model)
    _, allowed_actions, _ = vec_storm_simulator.reset()
    random_action = generate_uniform_action_given_mask(allowed_actions)
    random_action = np.array([random_action])
    for _ in range(max_steps):
        observations, vec_rewards, done, truncated, allowed_actions, metalabels = vec_storm_simulator.step(random_action)
        random_action = np.array([generate_uniform_action_given_mask(allowed_actions)])
        if "evade" in reward_model:
            assert vec_rewards[0] == 1.0 or vec_rewards[0] == 0.0, f"Reward is not 0 or 1: {vec_rewards[0]}"
        elif "geo" in reward_model:
            assert vec_rewards[0] == 1.0 or vec_rewards[0] == 0.0, f"Reward is not 0 or 1: {vec_rewards[0]}"
        elif "refuel-10" in reward_model:
            assert vec_rewards[0] == 1.0 or vec_rewards[0] == 0.0 or vec_rewards[0] == 3.0, f"Reward is not 0, 1 or 3: {vec_rewards[0]}"
    print("Reality of rewards tests passed for model with single simulator", reward_model)

def generate_uniform_action_given_mask_vectorized(mask):
    number_of_available_actions = sum(sum(mask))
    action = random.choice(range(number_of_available_actions))
    index = -1
    for i in range(len(mask[0])):
        if mask[0][i]:
            index += 1
        if index == action:
            return np.int32(i)
        
    return -1

def test_reality_of_rewards_multiple(vec_storm_simulator: vec_storm.StormVecEnv, storm_simulator, reward_model):
    num_envs = 8
    max_steps = 5000
    vec_storm_simulator.set_num_envs(num_envs)
    _, allowed_actions, _ = vec_storm_simulator.reset()
    random_actions = np.array([generate_uniform_action_given_mask(allowed_actions) for _ in range(num_envs)])
    
    for i in range(max_steps):
        observations, vec_rewards, done, truncated, allowed_actions, metalabels = vec_storm_simulator.step(random_actions)
        random_actions = np.array([generate_uniform_action_given_mask(allowed_actions) for _ in range(num_envs)])
        if i % 400 == 0:
            vec_storm_simulator.reset()
        if "evade" in reward_model or "geo" in reward_model:
            assert np.all(np.isin(vec_rewards, [0.0, 1.0])), f"Rewards contain values other than 0 or 1: {vec_rewards}"
        elif "refuel-10" in reward_model:
            assert np.all(np.isin(vec_rewards, [0.0, 1.0, 3.0])), f"Rewards contain values other than 0, 1, or 3: {vec_rewards}"
    
    print("Reality of rewards tests passed for model with multiple simulators", reward_model)


def perform_comparison(prism_model: str, prism_properties: str, constants: dict[str, str], reward_model: str = "evade"):
    print("Performing comparison on model", reward_model)
    
    model = initialize_prism_model(prism_model, prism_properties, constants)
    
    # Initialize storm simulator
    storm_simulator = simulator.create_simulator(model)
    
    labeling = model.labeling.get_labels()
    intersection_labels = [label for label in labeling if label in special_labels]
    vec_storm_simulator = vec_storm.StormVecEnv(model, get_scalarized_reward=get_scalarized_reward, num_envs=1, max_steps=400, metalabels={"goals": intersection_labels})

    storm_observation, rewards, labels = storm_simulator.restart()
    vec_observation_valuations, allowed_actions, goal_label_flags = vec_storm_simulator.reset()

    # Obligate tests
    storm_observation_valuation = create_valuations_encoding(storm_observation, model)
    assert vec_observation_valuations[0].tolist() == storm_observation_valuation.tolist(), f"Observations are different: {vec_observation_valuations[0]} vs {storm_observation_valuation}"
    test_number_of_available_actions_init(allowed_actions, storm_simulator.available_actions())

    print("Initialization tests passed")
    storm_rand_action = generate_uniform_action_given_range(storm_simulator.available_actions()) # How to generate random action with storm environment
    vec_rand_action = generate_uniform_action_given_mask(allowed_actions) # How to generate random action with vectorized environment

    
    print("Comparing overall performance for model", reward_model)
    compare_overall_performance(vec_storm_simulator, storm_simulator)

    test_reality_of_rewards_single(vec_storm_simulator, reward_model)
    test_reality_of_rewards_multiple(vec_storm_simulator, storm_simulator, reward_model)

    print("Tests finished")

if __name__ == "__main__":
    perform_comparison("./models/evade/sketch.templ", "./models/evade/sketch.props", "", "evade")
    perform_comparison("./models_large/geo-2-8/sketch.templ", "./models_large/geo-2-8/sketch.props", "", "geo")
    perform_comparison("./models/refuel-10/sketch.templ", "./models/refuel-10/sketch.props", "", "refuel-10")
