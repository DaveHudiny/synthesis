import os
import json

import tensorflow as tf


class ExtractionStats:
    def __init__(self, original_policy_reachability: float, original_policy_reward: float,
                 use_one_hot: bool = False, number_of_samples: int = 0, memory_size: int = 0,
                 residual_connection: bool = False):

        self.original_policy_reachability = original_policy_reachability
        self.original_policy_reward = original_policy_reward
        self.use_one_hots = use_one_hot
        self.memory_size = memory_size
        self.number_of_samples = number_of_samples
        self.residual_connection = residual_connection

        self.extracted_policy_reachabilities = []
        self.extracted_policy_rewards = []
        self.extracted_fsc_reachability = []
        self.extracted_fsc_reward = []

        self.evaluation_accuracies = []

    def add_extraction_result(self, extracted_policy_reachability: float, extracted_policy_reward: float):
        self.extracted_policy_reachabilities.append(
            extracted_policy_reachability)
        self.extracted_policy_rewards.append(extracted_policy_reward)

    def add_fsc_result(self, extracted_fsc_reachability: float, extracted_fsc_reward: float):
        self.extracted_fsc_reachability.append(extracted_fsc_reachability)
        self.extracted_fsc_reward.append(extracted_fsc_reward)
        
    def add_evaluation_accuracy(self, evaluation_accuracy: tf.Tensor):
        self.evaluation_accuracies.append(evaluation_accuracy.numpy())

    def store_as_json(self, model_name: str, experiments_path: str):
        if not os.path.exists(experiments_path):
            os.makedirs(experiments_path)
        index = 0
        while os.path.exists(os.path.join(experiments_path, f"{model_name}_extraction_stats_{index}.json")):
            index += 1
        with open(os.path.join(experiments_path, f"{model_name}_extraction_stats_{index}.json"), "w") as f:
            dictus = self.__dict__
            for key in dictus:
                dictus[key] = str(dictus[key])
            json.dump(dictus, f, indent=4)
