import tensorflow as tf
import numpy as np

from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy

from environment.environment_wrapper_vec import EnvironmentWrapperVec

from interpreters.direct_fsc_extraction.encoding_functions import get_encoding_functions


def compare_two_policies(policy1: TFPolicy, policy2: TFPolicy, buffer: TFUniformReplayBuffer, memory_encode, memory_decode, mem_size, environment : EnvironmentWrapperVec = None):
    policy1_eager = PyTFEagerPolicy(policy1, use_tf_function=True, batch_time_steps=False)
    policy2_eager = PyTFEagerPolicy(policy2, use_tf_function=True, batch_time_steps=False)
    batch_size = 8
    iterator = iter(buffer.as_dataset(sample_batch_size=batch_size, num_steps=25, num_parallel_calls=tf.data.AUTOTUNE, single_deterministic_pass=True).prefetch(tf.data.AUTOTUNE))
    policy_state1 = policy1.get_initial_state(batch_size)
    policy_state2 = policy2.get_initial_state(batch_size)
    
    for experience, _ in iterator:
        for i in range(len(experience.observation["observation"][0])):
            time_step = TimeStep(
                step_type=experience.step_type[:, i],
                reward=experience.reward[:, i],
                discount=experience.discount[:, i],
                observation={"observation": experience.observation["observation"][:, i], 
                             "mask": experience.observation["mask"][:, i], 
                             "integer" : experience.observation["integer"][:, i]}
            )
            policy_step1 = policy1_eager.action(time_step, policy_state1)
            policy_step2 = policy2_eager.action(time_step, policy_state2)
            action1 = policy_step1.action
            action2 = policy_step2.action
            policy_state1 = policy_step1.state
            policy_state2 = policy_step2.state
            for mem1, mem2 in zip(policy_state1.numpy(), policy_state2.numpy()):
                mem2 = memory_encode(mem_size, mem2).numpy()
                for mem1_val, mem2_val in zip(mem1, mem2):
                    assert mem1_val == mem2_val, f"Failed for {mem1_val}, {mem2_val}"
            for act1, act2 in zip(action1.numpy(), action2.numpy()):
                assert act1 == act2, f"Failed for {act1}, {act2}"

def test_memory_endoce_and_decode_functions(encode, decode, max_memory, memory_size):
    for i in range(max_memory):
        memory_encoded = encode(memory_size, i)
        memory_encoded = tf.reshape(memory_encoded, (1, memory_size))
        # print(memory_encoded)
        memory_decoded = decode(memory_size, memory_encoded)
        assert i == memory_decoded, f"Failed for {i}, {memory_decoded}"

def run_tests(args):
    memory_size = args.memory_size
    max_memory = 3 ** memory_size if not args.use_one_hot else args.memory_size
    encode, decode = get_encoding_functions(args.use_one_hot)
    test_memory_endoce_and_decode_functions(encode, decode, max_memory, memory_size)

