# Description: Random agent class that inherits from FatherAgent class. Not used.
# Author: David Hudák
# Login: xhudak03
# File: random_agent.py


import tensorflow as tf

from tf_agents.agents import TFAgent
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import random_tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

import logging
import tf_agents

from tools.encoding_methods import *

logger = logging.getLogger(__name__)

import collections
LossInfo = collections.namedtuple("LossInfo", ("loss", "extra"))

class RandomAgent(TFAgent):
    def __init__(self, time_step_spec, action_spec, *args, **kwargs):
        random_policy = random_tf_policy.RandomTFPolicy(time_step_spec, action_spec, observation_and_action_constraint_splitter=observation_and_action_constraint_splitter)
        super().__init__(
            time_step_spec,
            action_spec,
            policy=random_policy,
            collect_policy=random_policy,
            train_sequence_length=None
        )

    def _initialize(self):
        return tf.compat.v1.no_op()

    def _train(self, experience, weights):
        # Return Loss info with value 0.0
        return LossInfo(loss=tf.constant(0.0), extra={})