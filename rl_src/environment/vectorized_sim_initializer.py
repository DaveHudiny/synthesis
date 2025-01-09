# This file is used to initialize the vectorized simulation environment

import vec_storm
import os
import sys
import pickle as pkl
import re

import logging
logger = logging.getLogger(__name__)


class SimulatorInitializer:
    @staticmethod
    def load_and_store_simulator(stormpy_model, get_scalarized_reward, num_envs, max_steps, metalabels, model_path, compiled_models_path="compiled_models_vec_storm"):
        """ Load the simulator for the environment. If the model was not compiled previously, the model is compiled from scratch and saved. Otherwise, the model is loaded from the file.

        Args:
            stormpy_model (stormpy.model.Model): The stormpy model (POMDP).
            get_scalarized_reward (function): The function to get the scalarized reward.
            num_envs (int): The number of environments.
            max_steps (int): The maximum number of steps.
            metalabels (list): The metalabels of goal states.
            model_path (str): The path to the model. For loading and saving the model.
            compiled_models_path (str): The path to the compiled models.

        Returns:
            VectorizedSimulator: The simulator.
        """
        if not os.path.exists(compiled_models_path):
            os.makedirs(compiled_models_path)
        name = SimulatorInitializer.get_name_from_path(model_path)
        simulator = SimulatorInitializer.try_load_simulator_by_name_from_pickle(
            name, compiled_models_path)
        if simulator is None:
            logger.info(f"Compiling model {name}...")
            simulator = vec_storm.StormVecEnv(
                stormpy_model, get_scalarized_reward, num_envs=num_envs, max_steps=max_steps, metalabels=metalabels)
            simulator.save(f"{compiled_models_path}/{name}.pkl")
        return simulator

    @staticmethod
    def get_name_from_path(model_path):
        """ Get the name of the model from the model_path. The model path looks like /path/to/model/name/sketch.templ

        Args:
            path (str): The path to the model.

        Returns:
            str: The name of the model.
        """
        if model_path is None:
            return None
        return re.search(r"([^/]+)\/sketch.templ", model_path).group(1)

    @staticmethod
    def try_load_simulator_by_name_from_pickle(name, path_to_compiled_models):
        """ Try to load the simulator by name.

        Args:
            name (str): The name of the model.
            path_to_compiled_models (str): The path to the compiled models.

        Returns:
            VectorizedSimulator: The simulator.
        """
        if not os.path.exists(f"{path_to_compiled_models}/{name}.pkl") or name is None:
            logger.info(
                f"Model {name} not found in {path_to_compiled_models}. The model will be compiled.")
            return None
        else:
            logger.info(
                f"Model {name} found in {path_to_compiled_models}. The model will be loaded")
            try:
                return vec_storm.StormVecEnv.load(f"{path_to_compiled_models}/{name}.pkl")
            except Exception as e:
                logger.error(
                    f"Error while loading model {name} from {path_to_compiled_models}: {e}")
                return None
