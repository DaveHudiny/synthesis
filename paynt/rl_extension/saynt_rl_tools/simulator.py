import stormpy
import tensorflow as tf
import numpy as np

from stormpy.simulator import create_simulator

def init_simulator(pomdp_model, observation = None, state = None, belief = None):
    if observation != None: # Currently unknown state or belief
        indices = [i for i, x in enumerate(pomdp_model.observations) if x == observation]
        ones = np.ones((len(indices,)))
        logits = tf.math.log([ones])
        index = tf.random.categorical(logits, 1)
        index = tf.squeeze(index).numpy()
        indices_bitvector = stormpy.BitVector(pomdp_model.nr_states, [indices[index]])
    elif state != None:
        indices_bitvector = stormpy.BitVector(pomdp_model.nr_states, [state])
    elif belief != None:
        raise "Belief simulator initialization not implemented yet."
    else:
        index = pomdp_model.initial_states[0]
        indices_bitvector = stormpy.BitVector(pomdp_model.nr_states, [index])

    pomdp_model.set_initial_states(indices_bitvector)
    simulator = create_simulator(pomdp_model)
    return simulator