import json
import os


class ExtractionBenchmarkRes:
    def __init__(self, type : str, 
                 memory_size : int,
                 accuracies : list,
                 verified_performance : float,
                 original_rl_reward : float,
                 original_rl_reachability : float,
                 reachabilities : list,
                 rewards : list
                 ):
        self.type = type
        self.memory_size = memory_size
        self.accuracies = accuracies
        self.verified_performance = verified_performance
        self.original_rl_reward = original_rl_reward
        self.original_rl_reachability = original_rl_reachability
        self.reachabilities = reachabilities
        self.rewards = rewards

    def get_json(self):
        """
        Get the JSON representation of the object.
        """
        dictus_stringified = self.__dict__.copy()
        for key, value in dictus_stringified.items():
            dictus_stringified[key] = str(value)
        return json.dumps(dictus_stringified, indent=4)


class ExtractionBenchmarkResManager:
    @staticmethod
    def create_folder_with_extraction_benchmark_res(folder_path, list_of_extraction_benchmark_res : list[ExtractionBenchmarkRes]):
        """
        Create a folder with extraction benchmark results.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for i, extraction_benchmark_res in enumerate(list_of_extraction_benchmark_res):
            
            file_name = f"extraction_benchmark_res_{i}.json"
            while os.path.exists(os.path.join(folder_path, file_name)):
                i += 1
                file_name = f"extraction_benchmark_res_{i}.json"
            sub_folder_path = f"type_{extraction_benchmark_res.type}_memory_{extraction_benchmark_res.memory_size}"
            file_path = os.path.join(folder_path, sub_folder_path, file_name)
            if not os.path.exists(os.path.join(folder_path, sub_folder_path)):
                os.makedirs(os.path.join(folder_path, sub_folder_path))
            ExtractionBenchmarkResManager.save_extraction_benchmark_res(file_path, extraction_benchmark_res)
    
    @staticmethod
    def save_extraction_benchmark_res(file_path, extraction_benchmark_res : ExtractionBenchmarkRes):
        """
        Save extraction benchmark results to a file.
        """
        with open(file_path, 'w') as f:
            f.write(extraction_benchmark_res.get_json())
