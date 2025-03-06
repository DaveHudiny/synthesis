from rl_src.tools.encoding_methods import *
from paynt.quotient.fsc import FSC

import paynt.quotient.storm_pomdp_control as Storm_POMDP_Control
import paynt.quotient.pomdp as POMDP

import stormpy

from tf_agents.trajectories import StepType

import tf_agents.trajectories as Trajectories

from rl_src.environment import tf_py_environment

from paynt.quotient.fsc import FSC

import tensorflow as tf


from paynt.rl_extension.saynt_controller.simulation_controller import SAYNT_Simulation_Controller
from paynt.rl_extension.saynt_controller.saynt_step import SAYNT_Step
from paynt.rl_extension.saynt_controller.saynt_modes import SAYNT_Modes

import logging

logger = logging.getLogger(__name__)

import tqdm

class SAYNT_Driver:
    def __init__(self, observers: list = [], storm_control: Storm_POMDP_Control.StormPOMDPControl = None,
                 quotient: POMDP.PomdpQuotient = None, tf_action_labels: list = None,
                 encoding_method: EncodingMethods = EncodingMethods.VALUATIONS,
                 discount=0.99, fsc: FSC = None, q_values=None, model_reward_multiplier=-1.0):
        """Initialization of SAYNT driver.

        Args:
            observers (list, optional): List of callable observers, e.g. for adding data to replay buffers. Defaults to [].

        """
        assert storm_control is not None, "SAYNT driver needs Storm control with results"
        assert quotient is not None, "SAYNT driver needs quotient structure for model information"
        assert tf_action_labels is not None, "SAYNT driver needs action label indexing for proper functionality"
        self.quotient = quotient
        self.initial_pomdp_state = quotient.pomdp.initial_states[0]
        self.fsc = fsc
        self.observers = observers
        self.saynt_simulator = SAYNT_Simulation_Controller(
            storm_control, quotient, tf_action_labels, fsc=fsc, 
            paynt_q_values=q_values, model_reward_multiplier=model_reward_multiplier)
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

    def create_tf_time_step(self, saynt_step: SAYNT_Step) -> Trajectories.TimeStep:
        tf_saynt_step = Trajectories.TimeStep(step_type=tf.constant([saynt_step.tf_step_type]),
                                              reward=tf.constant([saynt_step.reward], tf.float32), discount=tf.constant([self.discount]),
                                              observation=tf.constant([create_valuations_encoding(saynt_step.observation, self.saynt_simulator.quotient.pomdp)]))
        return tf_saynt_step

    def create_tf_policy_step(self, saynt_step: SAYNT_Step) -> Trajectories.PolicyStep:
        #check whether the action is int or a tensor and convert it to tensor
        if type(saynt_step.action) == int:
            action = tf.constant([saynt_step.action])
        else: # reshape it to tensor with shhape (1,)
            action = tf.reshape(saynt_step.action, (1,))
        # same for fsc memory
        if type(saynt_step.fsc_memory) == int:
            fsc_memory = tf.constant([saynt_step.fsc_memory])
        else:
            fsc_memory = tf.reshape(saynt_step.fsc_memory, (1,))
        action = tf.cast(action, tf.int32)
        fsc_memory = tf.cast(fsc_memory, tf.int32)
        return Trajectories.PolicyStep(action, state=(), info=()) # TODO: info={"mem_node": fsc_memory}

    def episodic_run(self, episodes=5):
        cumulative_rewards = []
        episodes_finished_well = 0
        # Initialize TQDM progress bar
        tqdm_bar = tqdm.tqdm(total=episodes, desc="Episodic run")
        for _ in range(episodes):
            # saynt_step = self.saynt_simulator.reset()
            tqdm_bar.update(1)
            saynt_step = self.saynt_simulator.reset_belief_mdp()
            tf_saynt_step = self.create_tf_time_step(saynt_step)
            cumulative_reward = 0
            while saynt_step.tf_step_type != StepType.LAST:
                new_saynt_step = self.saynt_simulator.get_next_step(saynt_step)
                tf_policy_step = self.create_tf_policy_step(new_saynt_step)
                new_tf_saynt_step = self.create_tf_time_step(new_saynt_step)
                traj = Trajectories.from_transition(
                    tf_saynt_step, tf_policy_step, new_tf_saynt_step)
                saynt_step = new_saynt_step
                tf_saynt_step = new_tf_saynt_step
                cumulative_reward += (new_saynt_step.reward - new_saynt_step.virtual_reward)
                if new_saynt_step.virtual_reward > 0:
                    episodes_finished_well += 1
                for observer in self.observers:
                    observer(traj)
            cumulative_rewards.append(cumulative_reward)
            # print("Reward for episode:", cumulative_reward)
        print("Cumulative rewards mean:", np.mean(cumulative_rewards))
        print("Episodes finished well:", episodes_finished_well, "out of", episodes, "episodes")
        self.restore_pomdp_original_state()

    

    def step_run(self, steps=25):
        saynt_step = self.saynt_simulator.last_step # Recovery from previous simulation
        for _ in range(steps):
            if saynt_step == StepType.LAST:
                saynt_step = self.saynt_simulator.reset_belief_mdp()
            tf_saynt_step = self.create_tf_time_step(saynt_step)
            new_saynt_step = self.saynt_simulator.get_next_step(saynt_step)
            tf_policy_step = self.create_tf_policy_step(new_saynt_step)
            new_tf_saynt_step = self.create_tf_time_step(new_saynt_step)
            traj = Trajectories.from_transition(
                tf_saynt_step, tf_policy_step, new_tf_saynt_step)
            saynt_step = new_saynt_step
            tf_saynt_step = new_tf_saynt_step
            for observer in self.observers:
                observer(traj)
        print("Step runner performed for", steps, "steps")
            
    def restore_pomdp_original_state(self):
        nr_states = self.quotient.pomdp.nr_states
        indices_bitvector = stormpy.BitVector(nr_states, [self.initial_pomdp_state])
        self.quotient.pomdp.set_initial_states(indices_bitvector)
    # 251