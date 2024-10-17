from rl_src.tools.encoding_methods import *
from paynt.quotient.fsc import FSC

import paynt.quotient.storm_pomdp_control as Storm_POMDP_Control
import paynt.quotient.pomdp as POMDP

import stormpy

from tf_agents.trajectories import StepType

import tf_agents.trajectories as Trajectories

from tf_agents.environments import tf_py_environment

from paynt.quotient.fsc import FSC


from paynt.rl_extension.saynt_controller.simulation_controller import SAYNT_Simulation_Controller
from paynt.rl_extension.saynt_controller.saynt_step import SAYNT_Step
from paynt.rl_extension.saynt_controller.saynt_modes import SAYNT_Modes

import logging

logger = logging.getLogger(__name__)

class SAYNT_Driver:
    def __init__(self, observers: list = [], storm_control: Storm_POMDP_Control.StormPOMDPControl = None,
                 quotient: POMDP.PomdpQuotient = None, tf_action_labels: list = None,
                 encoding_method: EncodingMethods = EncodingMethods.VALUATIONS,
                 discount=0.99, fsc: FSC = None):
        """Initialization of SAYNT driver.

        Args:
            observers (list, optional): List of callable observers, e.g. for adding data to replay buffers. Defaults to [].

        """
        assert storm_control is not None, "SAYNT driver needs Storm control with results"
        assert quotient is not None, "SAYNT driver needs quotient structure for model information"
        assert tf_action_labels is not None, "SAYNT driver needs action label indexing for proper functionality"

        self.fsc = fsc
        self.observers = observers
        self.saynt_simulator = SAYNT_Simulation_Controller(
            storm_control, quotient, tf_action_labels, fsc=fsc)
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
        tf_saynt_step = Trajectories.TimeStep(step_type=saynt_step.tf_step_type,
                                              reward=saynt_step.reward, discount=self.discount,
                                              observation=create_valuations_encoding(saynt_step.observation, self.saynt_simulator.quotient.pomdp))
        return tf_saynt_step

    def create_tf_policy_step(self, saynt_step: SAYNT_Step) -> Trajectories.PolicyStep:
        return Trajectories.PolicyStep(saynt_step.action, state=(), info=())

    def episodic_run(self, episodes=5):
        for _ in range(episodes):
            # saynt_step = self.saynt_simulator.reset()
            saynt_step = self.saynt_simulator.reset_belief_mdp()
            tf_saynt_step = self.create_tf_time_step(saynt_step)
            cumulative_reward = 0
            while saynt_step.tf_step_type != StepType.LAST:
                tf_policy_step = self.create_tf_policy_step(saynt_step)
                new_saynt_step = self.saynt_simulator.get_next_step(saynt_step)
                new_tf_saynt_step = self.create_tf_time_step(new_saynt_step)
                traj = Trajectories.from_transition(
                    tf_saynt_step, tf_policy_step, new_tf_saynt_step)
                saynt_step = new_saynt_step
                tf_saynt_step = new_tf_saynt_step
                cumulative_reward += new_saynt_step.reward
                # for observer in self.observers:
                #     observer(traj)
            print("Celková odměna za epizodu:", cumulative_reward)

    def step_run(self, steps=25):
        pass