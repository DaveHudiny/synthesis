# This file contains the Synthesizer_RL class, which creates interface between RL and PAYNT.
# Author: David Hudák
# Login: xhudak03
# File: synthesizer_rl.py

from enum import Enum
from rl_src.environment.environment_wrapper import Environment_Wrapper
from rl_src.rl_main import ArgsEmulator, Initializer, save_statistics_to_new_json
from rl_src.interpreters.tracing_interpret import TracingInterpret
from rl_src.agents.policies.fsc_policy import FSC_Policy
from rl_src.tools.encoding_methods import *

import paynt.quotient.storm_pomdp_control as Storm_POMDP_Control
import paynt.quotient.pomdp as POMDP

import stormpy.storage as Storage


import tensorflow as tf
from tf_agents.trajectories import StepType

import tf_agents.trajectories as Trajectories

from tf_agents.environments import tf_py_environment

from paynt.quotient.fsc import FSC

import logging

logger = logging.getLogger(__name__)


class SAYNT_Modes(Enum):
    BELIEF = 1
    CUTOFF_FSC = 2
    CUTOFF_SCHEDULER = 3


class SAYNT_Step:
    """Class for step in SAYNT algorithm.
    """

    def __init__(self, action=0, observation = 0, state : Storage.SparseModelState = None, 
                 new_mode: SAYNT_Modes = SAYNT_Modes.BELIEF, tf_step_type : StepType = StepType.FIRST):
        """Initialization of the step.
        Args:
            action: Action.
            observation: Observation.
            reward: Reward.
            state: Storm state.
        """
        self.action = action
        self.observation = observation
        self.state = state
        self.new_mode = new_mode
        self.tf_step_type = tf_step_type


class SAYNT_Simulation_Controller:
    """Class for controller applicable in Storm simulator (or similar).
    """
    MODES = ["BELIEF", "Cutoff_FSC", "Scheduler"]

    def __init__(self, storm_control : Storm_POMDP_Control.StormPOMDPControl, quotient : POMDP.PomdpQuotient,
                 tf_action_labels : list = None, max_step_limit : int = 400):
        """Initialization of the controller.
        Args:
            storm_control: Result of the SAYNT algorithm.
            quotient: Important structure containing various information about the model etc.
            tf_action_labels: List of action labels. Index of action label correspond to output node.
        """
        self.storm_control = storm_control
        self.storm_control_result = storm_control.latest_storm_result
        self.quotient = quotient
        self.current_state = None
        self.current_mode = SAYNT_Modes.BELIEF
        self.tf_action_labels = tf_action_labels
        self.induced_mc_nr_states = self.storm_control_result.induced_mc_from_scheduler.nr_states
        self.max_step_limit = max_step_limit
        self.steps_performed = 0


        self.num_observations = quotient.pomdp.nr_observations
        self.get_choice_label = self.storm_control_result.induced_mc_from_scheduler.choice_labeling.get_labels_of_choice

    def get_next_step(self, prev_step : SAYNT_Step) -> SAYNT_Step:
        """Get the next action.
        Args:
            prev_step: Previous SAYNT_Step.
        Returns:
            SAYNT_Step: Next step.
        """
        
        self.current_state = prev_step
        self.current_mode = prev_step.new_mode
        self.steps_performed += 1
        if self.current_mode == SAYNT_Modes.BELIEF:
            return self.get_next_step_belief(prev_step)
        elif self.current_mode == SAYNT_Modes.CUTOFF_FSC:
            return self.get_next_action_cutoff_fsc(prev_step)
        elif self.current_mode == SAYNT_Modes.CUTOFF_SCHEDULER:
            return self.get_next_action_cutoff_scheduler(prev_step)
        else:
            raise ValueError("Unknown mode")
        
    def get_observations_and_action_from_labels(self, state: Storage.SparseModelState):
        observations = []
        actions = []
        for label in state.labels: # Is there really needed a for loop?
            if '[' in label:
                observation = self.quotient.observation_labels.index(label)
            elif 'obs_' in label:
                _, observation = label.split('_')
            else:
                continue
            choice_label = list(self.get_choice_label(state.id))[0]
            try:
                index = self.tf_action_labels.index(choice_label)
            except ValueError:
                index = -1
            observations.append(observation)
            actions.append(index)
        return observations[0], actions[0]

    def update_state(self, state : Storage.SparseModelState):
        """Function samples new state from transition matrix of induced MC given current state."""
        probs = self.storm_control.latest_storm_result.induced_mc_from_scheduler.transition_matrix[state.id]
        prob_row = np.zeros((self.induced_mc_nr_states))
        for prob_key in probs:
            prob_row[prob_key.column] = prob_key.value()
        logits = tf.math.log([prob_row])
        sample = tf.random.categorical(logits, num_samples=1)
        index = tf.squeeze(sample).numpy()
        new_state = self.storm_control.latest_storm_result.induced_mc_from_scheduler.states[index]
        return new_state
    
    def get_new_mode(self, state : Storage.SparseModelState):
        new_mode = None
        if "cutoff" not in state.labels and 'clipping' not in state.labels:
            new_mode = SAYNT_Modes.BELIEF
        else:
            if "finite_mem" in state.labels:
                new_mode = SAYNT_Modes.CUTOFF_FSC
            else:
                new_mode = SAYNT_Modes.CUTOFF_SCHEDULER
        return new_mode
    
    def get_tf_step_type(self, state):
        is_last = self.storm_control_result.induced_mc_from_scheduler.is_sink_state(state.id)
        if is_last or self.steps_performed > self.max_step_limit:
            return StepType.LAST
        else:
            return StepType.MID

    def get_next_step_belief(self, prev_step : SAYNT_Step) -> SAYNT_Step:
        """Get the next step in belief mode.
        Returns:
            SAYNT_Step: Next step.
        """
        state = self.update_state(prev_step.state)
        observation, action = self.get_observations_and_action_from_labels(prev_step.state)
        new_mode = self.get_new_mode(state)
        tf_step_type = self.get_tf_step_type(state)
        new_step = SAYNT_Step(action, observation, state, new_mode, tf_step_type)
        return new_step

    def get_next_action_cutoff_fsc(self, prev_step : SAYNT_Step) -> SAYNT_Step:
        """Get the next action in cutoff FSC mode.
        Returns:
            SAYNT_Step: Next step.
        """
        return prev_step

    def get_next_action_cutoff_scheduler(self, prev_step : SAYNT_Step) -> SAYNT_Step:
        """Get the next action in cutoff scheduler mode.
        Returns:
            SAYNT_Step: Next step.
        """
        return prev_step
    
    def reset(self) -> SAYNT_Step:
        """Resets the simulation with setting current state to initial state.
        Returns:
            SAYNT_Step: Initial state of induced MC.
        """
        self.steps_performed = 0
        init_states = self.storm_control_result.induced_mc_from_scheduler.initial_states
        n = len(init_states)
        index = tf.random.uniform(shape=[], minval=0, maxval=n, dtype=tf.int32)
        sample = tf.gather(init_states, index)
        state = self.storm_control_result.induced_mc_from_scheduler.states[sample]
        observation, action = self.get_observations_and_action_from_labels(state)
        mode = self.get_new_mode(state)
        return SAYNT_Step(action, observation, state, mode, StepType.FIRST)
    
class SAYNT_Driver:
    def __init__(self, observers : list = [], storm_control : Storm_POMDP_Control.StormPOMDPControl = None, 
                 quotient : POMDP.PomdpQuotient = None, tf_action_labels : list = None,
                 encoding_method : EncodingMethods = EncodingMethods.VALUATIONS,
                 discount = 0.99):
        """Initialization of SAYNT driver.

        Args:
            observers (list, optional): List of callable observers, e.g. for adding data to replay buffers. Defaults to [].
            
        """
        assert storm_control is not None, "SAYNT driver needs Storm control with results"
        assert quotient is not None, "SAYNT driver needs quotient structure for model information"
        assert tf_action_labels is not None, "SAYNT driver needs action label indexing for proper functionality"
        
        self.observers = observers
        self.saynt_simulator = SAYNT_Simulation_Controller(storm_control, quotient, tf_action_labels)
        self.encoding_method = encoding_method
        self.encoding_function = self.get_encoding_function(encoding_method)
        self.discount = discount
        
    def get_encoding_function(self, encoding_method):
        if encoding_method == EncodingMethods.VALUATIONS:
            return create_valuations_encoding
        elif encoding_method == EncodingMethods.ONE_HOT_ENCODING:
            return create_one_hot_encoding
        elif encoding_method == EncodingMethods.VALUATIONS_PLUS:
            return create_valuations_encoding_plus
        else:
            return (lambda x: [x])
        
    def create_tf_time_step(self, saynt_step : SAYNT_Step) -> Trajectories.TimeStep:
        tf_saynt_step = Trajectories.TimeStep(step_type=saynt_step.tf_step_type, 
                                              reward = 0, discount=self.discount,
                                              observation=create_valuations_encoding(saynt_step.observation, self.saynt_simulator.quotient.pomdp)) 
        return tf_saynt_step
    
    def create_tf_policy_step(self, saynt_step : SAYNT_Step) -> Trajectories.PolicyStep:
        return Trajectories.PolicyStep(saynt_step.action, state=(), info=())
    
    def episodic_run(self, episodes = 5):
        for _ in range(episodes):
            saynt_step = self.saynt_simulator.reset()
            tf_saynt_step = self.create_tf_time_step(saynt_step)
            while saynt_step.tf_step_type != StepType.LAST:
                tf_policy_step = self.create_tf_policy_step(saynt_step)
                new_saynt_step = self.saynt_simulator.get_next_step(saynt_step)
                new_tf_saynt_step = self.create_tf_time_step(new_saynt_step)
                traj = Trajectories.from_transition(tf_saynt_step, tf_policy_step, new_tf_saynt_step)
                saynt_step = new_saynt_step
                tf_saynt_step = new_tf_saynt_step
                # for observer in self.observers:
                #     observer(traj)
            
    
    def step_run(self, steps = 25):
        pass

class Synthesizer_RL:
    """Class for the interface between RL and PAYNT.
    """

    def __init__(self, stormpy_model, args: ArgsEmulator,
                 initial_fsc_multiplier: float = 1.0,
                 qvalues: list = None, action_labels_at_observation: dict = None):
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
        self.agent = self.initializer.initialize_agent(qvalues_table=qvalues, action_labels_at_observation=action_labels_at_observation)
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
        
    def get_encoding_method(self, method_str : str = "Valuations") -> EncodingMethods:
        if method_str == "Valuations":
            return EncodingMethods.VALUATIONS
        elif method_str == "One-Hot":
            return EncodingMethods.ONE_HOT_ENCODING
        elif method_str == "Valuations++":
            return EncodingMethods.VALUATIONS_PLUS
        else:
            return EncodingMethods.INTEGER
        
    def get_saynt_trajectories(self, storm_control, quotient):
        observer = self.agent.replay_buffer.add_batch
        tf_action_labels = self.initializer.environment.action_keywords
        if not hasattr(self, "saynt_driver"):
            encoding_method = self.get_encoding_method(self.initializer.args.encoding_method)
            self.saynt_driver = SAYNT_Driver([observer], storm_control, quotient, 
                                             tf_action_labels, encoding_method, self.initializer.args.discount_factor)
        self.saynt_driver.episodic_run(5)
