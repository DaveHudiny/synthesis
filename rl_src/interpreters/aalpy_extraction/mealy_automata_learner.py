import aalpy

import pickle
from tf_agents.trajectories import time_step as ts
from collections import defaultdict
from tf_agents.trajectories import StepType
from aalpy.learning_algs import run_RPNI
from aalpy.automata import MooreMachine,MealyMachine

import numpy as np

class MealyAutomataLearner:
    """
    This class is responsible for learning automata from a given set of traces.
    """

    @staticmethod
    def evaluate_valiation_accuracy(validation_traces, automata : MealyMachine):
        """
        Evaluates the validation accuracy of the learned automata on the given traces.
        
        Args:
            validation_traces (list): A list of traces to be used for validation.
            automata (MealyMachine): The learned Mealy machine.
        
        Returns:
            float: The accuracy of the automata on the validation traces.
        """
        pass

    @staticmethod
    def compute_episode_ranges(step_types) -> list:
        episodes_starts = np.argwhere(step_types == StepType.FIRST) # Indices in form of (batch, time)
        episodes_ends = np.argwhere(step_types == StepType.LAST) # Indices in form of (batch, time)
        episodes_starts = np.concatenate((episodes_starts, np.arange(episodes_starts.shape[0]).reshape(-1, 1)), axis=1)
        
        episode_ends_nums_by_batches = np.bincount(episodes_ends[:,0], minlength=step_types.shape[0])
        episode_starts_nums_by_batches = np.bincount(episodes_starts[:,0], minlength=step_types.shape[0])
        overflows = episode_starts_nums_by_batches - episode_ends_nums_by_batches
        overflow_indices = np.argwhere(overflows > 0)
        overflow_cumsums = np.cumsum(overflows) - 1
        num_overflows = len(overflow_indices)
        starts_for_removal = np.zeros((num_overflows,), dtype=int)
        last_start_step = [0, 0]
        for start in episodes_starts:
            if start[0] > last_start_step[0]:
                if last_start_step[0] in overflow_indices:
                    starts_for_removal[overflow_cumsums[last_start_step[0]]] = last_start_step[2]
            last_start_step = start
        if last_start_step[0] in overflow_indices:
            starts_for_removal[overflow_cumsums[last_start_step[0]]] = last_start_step[2]
        
        # Remove the overflows
        episodes_starts = np.delete(episodes_starts, starts_for_removal, axis=0)
        episodes_starts = episodes_starts[:, :-1]
        episode_ranges = np.concatenate((episodes_starts, episodes_ends[:, -1].reshape(-1, 1)), axis=1)
        return episode_ranges

    @staticmethod
    def convert_trajectories_to_episodes(trajectories) -> tuple[list, list]:
        """
        Converts the given trajectories into a list of episodes.
        
        Args:
            trajectories (list): A list of trajectories to be converted.
        
        Returns:
            tuple[list, list]: A tuple of episodes observations and episodes actions.
        """
        observations : np.ndarray = trajectories.observation.numpy()
        actions : np.ndarray = trajectories.action.numpy()
        step_types : np.ndarray = trajectories.step_type.numpy()
        
        episode_ranges = MealyAutomataLearner.compute_episode_ranges(step_types)
        
        # Indices of overflows
        
        episodes_observations = [observations[episode_range[0], episode_range[1]:episode_range[2]] for episode_range in episode_ranges]
        episodes_actions = [actions[episode_range[0], episode_range[1]:episode_range[2]] for episode_range in episode_ranges]
        return episodes_observations, episodes_actions


    @staticmethod
    def extract_mealy_machine(traces):
        """
        Extracts a Mealy machine from the given traces.
        
        Args:
            traces (list): A list of traces to be used for learning the automata.
        
        Returns:
            tuple: A tuple containing the learned Mealy machine and its action labels.
        """
        MealyAutomataLearner.convert_trajectories_to_episodes(traces)



        