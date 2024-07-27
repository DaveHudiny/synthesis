# This file contains the Synthesizer_RL class, which creates interface between RL and PAYNT.
# Author: David Hudák
# Login: xhudak03
# File: synthesizer_rl.py

from enum import Enum
from rl_src.environment.environment_wrapper import Environment_Wrapper
from rl_src.rl_main import ArgsEmulator, Initializer, save_statistics_to_new_json
from rl_src.interpreters.tracing_interpret import TracingInterpret
from rl_src.agents.policies.fsc_policy import FSC_Policy


import tensorflow as tf

from tf_agents.environments import tf_py_environment

from paynt.quotient.fsc import FSC

import logging

logger = logging.getLogger(__name__)


class SAYNT_Modes(Enum):
    BELIEF = 1
    CUTOFF_FSC = 2
    CUTOFF_SCHEDULER = 3


class SAYNT_STEP:
    """Class for step in SAYNT algorithm.
    """

    def __init__(self, action=0, memory_update=0, new_mode: SAYNT_Modes = SAYNT_Modes.BELIEF):
        """Initialization of the step.
        Args:
            action: Action.
            observation: Observation.
            reward: Reward.
            next_state: Next state.
        """
        self.action = action
        self.memory_update = memory_update
        self.new_mode = new_mode


class SAYNT_Simulation_Controller:
    """Class for controller applicable in Storm simulator (or similar).
    """
    MODES = ["BELIEF", "Cutoff_FSC", "Scheduler"]

    def __init__(self, saynt_result, num_observations: int = None, observation_labels=None, action_labels=None):
        """Initialization of the controller.
        Args:
            saynt_result: Result of the SAYNT algorithm.
        """
        self.saynt_result = saynt_result
        self.current_state = None
        self.current_mode = SAYNT_Modes.BELIEF

        self.num_observations = num_observations
        self.observation_labels = observation_labels
        self.action_labels = action_labels

    def get_next_action(self, state):
        """Get the next action.
        Args:
            state: Current state.
        Returns:
            str: Next action.
        """
        self.current_state = state
        if self.current_mode == SAYNT_Modes.BELIEF:
            return self.get_next_action_belief()
        elif self.current_mode == SAYNT_Modes.CUTOFF_FSC:
            return self.get_next_action_cutoff_fsc()
        elif self.current_mode == SAYNT_Modes.CUTOFF_SCHEDULER:
            return self.get_next_action_cutoff_scheduler()
        else:
            raise ValueError("Unknown mode")

    def get_next_action_belief(self):
        """Get the next action in belief mode.
        Returns:
            str: Next action.
        """
        pass

    def get_next_action_cutoff_fsc(self):
        """Get the next action in cutoff FSC mode.
        Returns:
            str: Next action.
        """
        pass

    def get_next_action_cutoff_scheduler(self):
        """Get the next action in cutoff scheduler mode.
        Returns:
            str: Next action.
        """
        pass


class Synthesizer_RL:
    """Class for the interface between RL and PAYNT.
    """

    def __init__(self, stormpy_model, args: ArgsEmulator,
                 initial_fsc_multiplier: float = 1.0,
                 qvalues: list = None):
        """Initialization of the interface.
        Args:
            stormpy_model: Model of the environment.
            args (ArgsEmulator): Arguments for the initialization.
            initial_fsc_multiplier (float, optional): Initial soft FSC multiplier. Defaults to 1.0.
        """

        self.initializer = Initializer(args, stormpy_model)
        self.initializer.environment = Environment_Wrapper(
            self.initializer.pomdp_model, args)
        self.initializer.tf_environment = tf_py_environment.TFPyEnvironment(
            self.initializer.environment)
        logger.info("RL Environment initialized")
        self.agent = self.initializer.initialize_agent(qvalues_table=qvalues)
        self.interpret = TracingInterpret(self.initializer.environment, self.initializer.tf_environment,
                                          self.initializer.args.encoding_method,
                                          possible_observations=self.initializer.environment._possible_observations)
        self.fsc_multiplier = initial_fsc_multiplier

    def train_agent(self, iterations: int):
        """Train the agent.
        Args:
            iterations (int): Number of iterations.
        """
        self.agent.train_agent_off_policy(iterations)
        self.agent.save_agent()

    def interpret_agent(self, best: bool = False, with_refusing: bool = False, greedy: bool = False):
        """Interpret the agent.
        Args:
            best (bool, optional): Whether to use the best, or the last trained agent. Defaults to False.
            with_refusing (bool, optional): Whether to use refusing. Defaults to False.
            greedy (bool, optional): Whether to use greedy policy evaluation (in case of PPO). Defaults to False.
        Returns:
            dict: Dictionary of the interpretation.
        """
        self.agent.load_agent(best)
        # Works only with agents which use policy wrapping (in our case only PPO)
        if greedy:
            self.agent.set_agent_stochastic()
        else:
            self.agent.set_agent_greedy()
        return self.interpret.get_dictionary(self.initializer.agent, with_refusing)

    def update_fsc_multiplier(self, multiplier: float):
        """Multiply the FSC multiplier.
        Args:
            multiplier (float): Multiplier for multiplication.
        """
        self.fsc_multiplier *= multiplier

    def train_agent_with_fsc_data(self, iterations: int, fsc: FSC, soft_decision: bool = False):
        """Train the agent with FSC data.
        Args:
            iterations (int): Number of iterations.
            fsc (FSC): FSC data.
            soft_decision (bool, optional): Whether to use soft decision. Defaults to False.
        """
        try:
            self.agent.load_agent()
        except:
            logger.info("Agent not loaded, training from scratch.")
        self.agent.init_fsc_policy_driver(
            self.initializer.tf_environment, fsc, soft_decision, self.fsc_multiplier)
        self.agent.train_agent_off_policy(iterations)

    def train_agent_combined_with_fsc(self, iterations: int, fsc: FSC):
        """Train the agent combined with FSC. Deprecated.
        Args:
            iterations (int): Number of iterations.
            fsc (FSC): FSC data.
        """
        assert self.initializer.args.learning_method == "PPO", "FSC policy can be created only for PPO agent"
        self.agent.wrapper._set_fsc_oracle(
            fsc, self.initializer.environment.action_keywords)
        self.agent.init_collector_driver(
            self.initializer.tf_environment)
        self.agent.train_agent_off_policy(iterations)

    def save_to_json(self, experiment_name: str = "PAYNTc+RL"):
        """Save the agent to JSON.
        Args:
            experiment_name (str): Name of the experiment.
        """
        evaluation_result = self.agent.evaluation_result
        save_statistics_to_new_json(
            experiment_name, "model", "PPO", evaluation_result, self.initializer.args)
