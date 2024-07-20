# Implementation of PPO with critic represetned by Q-values obtained from FSC.
# Author: David Hudák
# Login: xhudak03
# File: ppo_with_qvalues_fsc.py

from agents.father_agent import FatherAgent

from tf_agents.networks import network

import sys
sys.path.append("../")

import paynt.parser.sketch
import paynt.synthesizer.synthesizer_pomdp



class FSC_Critic(network.Network):
    def __init__(self, input_tensor_spec, name="FSC_QValue_Estimator"):
        
        super(FSC_Critic, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name=name)
        
        
    

class PPO_with_QValues_FSC(FatherAgent):
    def __init__(self, environment, tf_environment, args, load=False, agent_folder=None):
        self.common_init(environment, tf_environment, args, load, agent_folder)
        self.agent = None
        self.policy_state = None
        
        sketch_path = args.prism_model
        props_path = args.prism_properties
        qvalues_function = self.compute_qvalues_function(sketch_path, props_path)

    def get_evaluation_policy(self):
        return self.agent.collect_policy

    def get_initial_state(self, batch_size=None):
        return self.agent.collect_policy.get_initial_state(batch_size)

    def save_agent(self, best=False):
        self.agent.save()

    def load_agent(self, best=False):
        self.agent.load()
        
    def compute_qvalues_function(self, sketch_path, properties_path):
        quotient = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)
        k = 3 # May be unkown?
        quotient.set_imperfect_memory_size(k)
        synthesizer = paynt.synthesizer.synthesizer_pomdp.SynthesizerPomdp(quotient, method="ar", storm_control=None)
        assignment = synthesizer.synthesize()

        # before the quotient is modified we can use this assignment to compute Q-values
        assert assignment is not None
        qvalues = quotient.compute_qvalues(assignment)

        # note Q(s,n) may be None if (s,n) exists in the unfolded POMDP but is not reachable in the induced DTMC
        memory_size = len(qvalues[0])
        assert k == memory_size
        for state in range(quotient.pomdp.nr_states):
            for memory in range(memory_size):
                print(f"s = {state}, n = {memory}, Q(s,n) = {qvalues[state][memory]}")