import tensorflow as tf

# from tools.evaluators import EvaluationResults

class SpecificationChecker:
    def __init__(self, optimization_specification = "reachability", optimization_goal = "max", evaluation_results = None):
        self.optimization_specification = optimization_specification
        self.optimization_goal = optimization_goal
        self.evaluation_results = evaluation_results
        if evaluation_results is not None:
            self.current_optimal_value = self.get_optimal_extractable_value(evaluation_results, optimization_specification, optimization_goal)

    def get_optimal_extractable_value(self, evaluation_result, specification_goal : str, optimization_goal : str) -> float:
        if specification_goal == "reachability":
            return evaluation_result.reach_probs[-1]
        elif specification_goal == "reward":
            return evaluation_result.returns[-1]
        else:
            raise ValueError("Unknown specification goal")
        
    def set_optimal_value(self, value : float):
        self.current_optimal_value = value
        
    def check_specification(self, evaluation_results) -> bool:
        if self.optimization_specification == "reachability":
            if self.optimization_goal == "max":
                return evaluation_results.reach_probs[-1] >= self.current_optimal_value
            elif self.optimization_goal == "min":
                return evaluation_results.reach_probs[-1] <= self.current_optimal_value
            else:
                raise ValueError("Unknown optimization goal")
        elif self.optimization_specification == "reward":
            if self.optimization_goal == "max":
                return evaluation_results.returns[-1] >= self.current_optimal_value
            elif self.optimization_goal == "min":
                return evaluation_results.returns[-1] <= self.current_optimal_value
            else:
                raise ValueError("Unknown optimization goal")
        else:
            raise ValueError("Unknown optimization specification")