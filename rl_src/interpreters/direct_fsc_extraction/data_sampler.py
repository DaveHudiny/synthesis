import tf_agents

from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.environments.tf_py_environment import TFPyEnvironment

from environment.environment_wrapper_vec import EnvironmentWrapperVec


def sample_data_with_policy(policy: TFPolicy, num_samples=100,
                            environment: EnvironmentWrapperVec = None,
                            tf_environment: TFPyEnvironment = None) -> TFUniformReplayBuffer:
    prev_time_step = tf_environment.reset()
    policy_state = policy.get_initial_state(environment.batch_size)
    replay_buffer = TFUniformReplayBuffer(
        data_spec=policy.trajectory_spec, batch_size=environment.batch_size, max_length=num_samples+1)
    action_function = tf_agents.utils.common.function(policy.action)

    for i in range(num_samples):
        policy_step = action_function(prev_time_step, policy_state)
        action = policy_step.action
        policy_state = policy_step.state
        time_step = tf_environment.step(action)
        traj = tf_agents.trajectories.trajectory.from_transition(
            prev_time_step, policy_step, time_step)
        replay_buffer.add_batch(traj)
        prev_time_step = time_step
    return replay_buffer
