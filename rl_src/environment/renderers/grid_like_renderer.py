from vec_storm.storm_vec_env import StormVecEnv

from environment.renderers.mba_renderer import MBARenderer

import stormpy.pomdp

LIST_OF_MODELS = ["mba", "mba-small", "drone-2-6-1", "drone-2-8-1", "geo-2-8", 
                             "refuel-10", "refuel-20", "intercept", "super-intercept", "evade", 
                             "rocks-16", "rocks-4-20"]

class GridLikeRenderer:
    def __init__(self, vec_storm_env : StormVecEnv, model_name : str):
        if model_name not in LIST_OF_MODELS:
            raise ValueError(f"Model name {model_name} not found in available models")
        self.vec_storm_env = vec_storm_env
        self.model_name = model_name
        self.initialize_grid_matrix(model_name)

    def initialize_grid_matrix(self, model_name):
        if model_name == "mba":
            self._renderer = MBARenderer(self.vec_storm_env)
        else:
            raise ValueError(f"Model name {model_name} not found in available models")
        # Implements initialization of the grid matrix for the environment
        
    def plot_slideshow_grid(self, trajectory):
        # Implements plotting of the slideshow grid for the environment
        self._renderer.plot_slideshow_grid(trajectory=trajectory)

    def get_trajectory_data(self):
        # Implements getting the trajectory data for the environment
        return self._renderer.get_trajectory_data()

    def render(self, mode="rgb_array", trajectory = None):
        # Implements rendering of the environment using the preinitialized grid matrix
        if mode == "human":
            return self.plot_slideshow_grid(trajectory)
        elif mode == "rgb_array":
            return self.get_trajectory_data()