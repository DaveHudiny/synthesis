import tensorflow as tf
from tf_agents.networks import value_rnn_network, actor_distribution_network
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer

from tf_agents.environments import TFPyEnvironment
from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

from rl_src.agents.networks.actor_networks import create_recurrent_actor_net_demasked
from rl_src.agents.networks.value_networks import create_recurrent_value_net_demasked
from tf_agents.networks.actor_distribution_rnn_network import ActorDistributionRnnNetwork
from tf_agents.policies.actor_policy import ActorPolicy
from tf_agents.networks.value_rnn_network import ValueRnnNetwork

from rl_src.tools.args_emulator import ArgsEmulator
from rl_src.agents.policies.fsc_policy import FSC_Policy
from rl_src.agents.policies.policy_mask_wrapper import Policy_Mask_Wrapper
from rl_src.environment.environment_wrapper import Environment_Wrapper
from rl_src.tools.encoding_methods import observation_and_action_constraint_splitter, observation_and_action_constraint_splitter_no_mask
from rl_src.tools.evaluators import *

from tf_agents.trajectories import Trajectory
from tf_agents.trajectories import from_transition
from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy
from tf_agents.agents.tf_agent import TFAgent
from tf_agents.specs import TensorSpec

from paynt.quotient.fsc import FSC




class Actor_Value_Pretrainer:
    def __init__(self, environment: Environment_Wrapper_Vec, tf_environment: TFPyEnvironment, args: ArgsEmulator, collect_data_spec=None):
        self.args = args

        self.environment = environment
        self.tf_environment = tf_environment

        self.actor_net = create_recurrent_actor_net_demasked(
            tf_environment, tf_environment.action_spec())
        self.critic_net = create_recurrent_value_net_demasked(tf_environment)

        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0005, weight_decay=0.00001)
        self.actor_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        self.gamma = 0.99

        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0005, weight_decay=0.00001)
        self.critic_loss_fn = tf.keras.losses.MeanSquaredError()
        self.init_replay_buffer(
            tf_environment, collect_data_spec=collect_data_spec)
        self.evaluation_result = EvaluationResults(self.environment.goal_value)

    def init_replay_buffer(self, tf_environment: TFPyEnvironment, collect_data_spec=None, buffer_size=None):
        """Initialize the uniform replay buffer for the agent.

        Args:
            tf_environment: The TensorFlow environment object, used for providing important specifications.
        """
        if buffer_size is None:
            buffer_size = self.args.buffer_size
        modified_collect_data_spec = collect_data_spec._replace(
            policy_info={"mem_node": TensorSpec(shape=(1, ), dtype=tf.int32, name='mem_node')})
        self.replay_buffer = TFUniformReplayBuffer(
            data_spec=modified_collect_data_spec,
            batch_size=tf_environment.batch_size,
            max_length=500000)

    def init_fsc_policy_driver(self, tf_environment: TFPyEnvironment, fsc: FSC = None, info_spec=None):
        """Initialize the FSC policy driver for the agent. Used for imitation learning with FSC.

        Args:
            tf_environment: The TensorFlow environment object, used for simulation information.
            fsc: The FSC object for imitation learning.
            soft_decision: Whether to use soft decision for FSC. Used only in PPO initialization.
            fsc_multiplier: The multiplier for the FSC. Used only in PPO initialization.
        """
        self.fsc = fsc
        self.fsc_policy = FSC_Policy(tf_environment=tf_environment, fsc=fsc,
                                     observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
                                     tf_action_keywords=self.environment.action_keywords,
                                     info_spec={"mem_node": TensorSpec(
                                         shape=(1, ), dtype=tf.int32, name='mem_node')},
                                     info_mem_node=True)
        eager = PyTFEagerPolicy(
            self.fsc_policy, use_tf_function=True, batch_time_steps=False)
        observers = [self.get_demasked_observer()]
        self.fsc_driver = DynamicEpisodeDriver(
            tf_environment,
            eager,
            observers=observers,
            num_episodes=64
        )

    def get_demasked_observer(self):
        """Observer for replay buffer. Used to demask the observation in the trajectory. Used with policy wrapper."""
        def _add_batch(item: Trajectory):
            modified_item = Trajectory(
                step_type=item.step_type,
                observation=item.observation["observation"],
                action=item.action,
                policy_info=(item.policy_info),
                next_step_type=item.next_step_type,
                reward=item.reward,
                discount=item.discount,
            )
            self.replay_buffer._add_batch(modified_item)
        return _add_batch

    def train_actor_iteration(self, actor_net: ActorDistributionRnnNetwork, experience):
        observations, actions, step_types = experience.observation, experience.action, experience.step_type
        with tf.GradientTape() as tape:
            predicted_actions, network_state = actor_net(
                observations, step_type=step_types)
            logits = predicted_actions.logits_parameter()
            _, _, num_classes = logits.shape
            actions = tf.reshape(actions, (-1,))
            logits = tf.reshape(logits, (-1, num_classes))
            actor_loss = self.actor_loss_fn(actions, logits)

        
        grads = tape.gradient(actor_loss, actor_net.trainable_variables)
        grads = [tf.clip_by_norm(grad, clip_norm=1.0) for grad in grads]
        self.actor_optimizer.apply_gradients(
            zip(grads, actor_net.trainable_variables))

        return actor_loss

    def train_value_iteration(self, critic_net: ValueRnnNetwork, experience):
        observations, rewards, step_types = experience.observation, experience.reward, experience.step_type
        current_observations = observations[:, :-1, :]
        next_observations = observations[:, 1:, :]
        rewards = rewards[:, :-1]
        predicted_values, _ = critic_net(
            current_observations, step_type=step_types[:, :-1])
        predicted_next_values, _ = critic_net(
            next_observations, step_type=step_types[:, 1:])
        returns = rewards + self.gamma * \
            tf.stop_gradient(predicted_next_values)
        with tf.GradientTape() as tape2:
            predicted_values, _ = critic_net(
                current_observations, step_type=step_types[:, :-1])
            critic_loss = self.critic_loss_fn(returns, predicted_values)

        grads = tape2.gradient(critic_loss, critic_net.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(grads, critic_net.trainable_variables))

        return critic_loss

    def fill_replay_buffer_with_fsc(self, fsc: FSC = None):
        if fsc is not None:
            self.reinit_fsc_policy_driver(fsc)
        assert self.fsc_driver is not None, "FSC driver should be pre-initialized or provided with function call."
        self.fsc_driver.run()

    def reinit_fsc_policy_driver(self, fsc: FSC = None):
        self.init_fsc_policy_driver(
            tf_environment=self.tf_environment, fsc=fsc)

    def train_both_networks(self, num_epochs: int, fsc: FSC, external_actor_net: ActorDistributionRnnNetwork = None, external_critic_net: ValueRnnNetwork = None,
                            use_best_traj_only = False, offline_data = False):
        if external_actor_net is not None:
            actor_net = external_actor_net
        else:
            actor_net = self.actor_net

        if external_critic_net is not None:
            critic_net = external_critic_net
        else:
            critic_net = self.critic_net

        if fsc is not None:
            self.reinit_fsc_policy_driver(fsc=fsc)
        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=8, sample_batch_size=256, num_steps=32, single_deterministic_pass=False).prefetch(4)
        self.iterator = iter(dataset)

        if use_best_traj_only:
            observer = self.get_demasked_observer()
            runner = self.get_separator_driver_runner(self.fsc, observers=[observer])

        if not offline_data:
            for _ in range(num_epochs):
                if use_best_traj_only:
                    runner(True, 2)
                self.fsc_driver.run()
        for epoch in range(num_epochs):
            # for _ in range(5):
            #     if use_best_traj_only:
            #         runner(True, 5)
            #     else:
            self.fsc_driver.run()

            experience, _ = next(self.iterator)

            actor_loss = self.train_actor_iteration(
                actor_net, experience=experience)
            critic_loss = self.train_value_iteration(
                critic_net, experience=experience)

            if epoch % 10 == 0:
                if epoch % 50 == 0:
                    self.evaluate_actor(actor_net, 40)
                print(f"Epoch: {epoch}, Actor Loss: {actor_loss.numpy()}")
                print(f"Epoch: {epoch}, Critic Loss: {critic_loss.numpy()}")
                

    def get_separator_driver_runner(self, fsc, observers):
        self.fsc_policy = FSC_Policy(tf_environment=self.tf_environment, fsc=fsc,
                                     observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
                                     tf_action_keywords=self.environment.action_keywords,
                                     info_spec={"mem_node": TensorSpec(
                                         shape=(1, ), dtype=tf.int32, name='mem_node')},
                                     info_mem_node=True)
        eager = PyTFEagerPolicy(
            self.fsc_policy, use_tf_function=True, batch_time_steps=False)
        
        def runner(run_until_success = True, max_episodes = 10):
            time_step = self.tf_environment.reset()
            policy_state = eager.get_initial_state(None)
            success = False
            episodes_performed = 0
            while not success and episodes_performed < max_episodes:
                trajectory_buffer = []
                while not time_step.is_last():
                    policy_step = eager.action(time_step, policy_state)
                    old_time_step = time_step
                    time_step = self.tf_environment.step(policy_step.action)
                    policy_state = policy_step.state
                    trajectory = from_transition(old_time_step, policy_step, time_step)
                    trajectory_buffer.append(trajectory)
                if run_until_success and not self.environment.flag_goal:
                    success = False
                else:
                    success = True
                episodes_performed += 1
            for item in trajectory_buffer:
                for observer in observers:
                    observer(item)
        
        return runner

    def get_duplex_driver(self, fsc: FSC, rl_agent: TFAgent, replay_buffer_fsc: TFUniformReplayBuffer, replay_buffer_rl: TFUniformReplayBuffer,
                          parallel_policy: Policy_Mask_Wrapper):
        observer = self.get_duplex_observer(
            replay_buffer_fsc=replay_buffer_fsc, replay_buffer_rl=replay_buffer_rl)
        fsc_policy = FSC_Policy(tf_environment=self.tf_environment, fsc=fsc,
                                observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
                                tf_action_keywords=self.environment.action_keywords, info_spec=rl_agent.collect_policy.info_spec,
                                parallel_policy=parallel_policy, switch_probability=0.05, duplex_buffering=True)

        eager = PyTFEagerPolicy(
            fsc_policy, use_tf_function=True, batch_time_steps=False)
        duplex_driver = DynamicEpisodeDriver(
            self.tf_environment, eager, observers=[observer], num_episodes=1)
        return duplex_driver

    def create_actor_evaluator_runner(self, actor_net: ActorDistributionRnnNetwork):
        policy = tf.function(actor_net.call)

        def runner(tf_environment: TFPyEnvironment, environment: Environment_Wrapper):
            time_step = tf_environment.reset()
            policy_state = actor_net.get_initial_state(1)
            cumulative_return = 0.0
            goal_visited = False

            while not time_step.is_last():
                observation = time_step.observation["observation"]
                step_type = time_step.step_type
                action, policy_state = policy(
                    observation, step_type=step_type, network_state=policy_state)
                time_step = tf_environment.step(action.sample())
                cumulative_return += time_step.reward / environment.normalizer

            if environment and environment.flag_goal:
                goal_visited = True

            return cumulative_return, goal_visited

        return runner
    
    def init_vectorized_evaluation_driver(self, tf_environment: tf_py_environment.TFPyEnvironment, environment: Environment_Wrapper_Vec, num_steps=400, actor_net = None):
        """Initialize the vectorized evaluation driver for the agent. Used for evaluation of the agent.

        Args:
            tf_environment: The TensorFlow environment object, used for simulation information.
            environment: The vectorized environment object, used for simulation information.
            num_steps: The number of steps for evaluation.
        """
        self.trajectory_buffer = TrajectoryBuffer(environment)
        policy = ActorPolicy(tf_environment.time_step_spec(), tf_environment.action_spec(), actor_net, 
                             observation_and_action_constraint_splitter=observation_and_action_constraint_splitter_no_mask)
        eager = PyTFEagerPolicy(
            policy=policy, use_tf_function=True, batch_time_steps=False)
        self.vec_driver = DynamicStepDriver(
            tf_environment,
            eager,
            observers=[self.trajectory_buffer.add_batched_step],
            num_steps=(1 + num_steps) * self.args.num_environments
        )

    def evaluate_actor(self, actor_net: ActorDistributionRnnNetwork, num_episodes: int):
        self.environment.set_random_starts_simulation(False)
        if self.args.vectorized_envs:
            if not hasattr(self, "vec_driver"):
                self.init_vectorized_evaluation_driver(
                    self.tf_environment, self.environment, num_steps=400, actor_net=actor_net)
            self.vec_driver.run()
            self.trajectory_buffer.final_update_of_results(
                self.evaluation_result.update)
            self.trajectory_buffer.clear()
            log_evaluation_info(self.evaluation_result)
        else:
            
            avg_return, avg_episodic_return, success_rate = compute_average_return(policy=None, tf_environment=self.tf_environment, 
                                                                                   num_episodes=num_episodes, environment=self.environment,
                                                                                   custom_runner=self.create_actor_evaluator_runner(actor_net))
        
        
            print("Average Return =", avg_return)
            print("Average Virtual Goal Value =", avg_episodic_return)
            print("Goal Reach Probability =", success_rate)

    def evaluate_critic(self, critic_net):
        pass

    def get_duplex_observer(self, replay_buffer_fsc: TFUniformReplayBuffer, replay_buffer_rl: TFUniformReplayBuffer):
        """Observer for two replay buffers. Used to demask the observation in the trajectory and select proper replay buffer.
            Used to exclude data generated by FSC from the other data obtained with RL policy to do separate training in future.
        """
        def _add_batch(item: Trajectory):
            if item.policy_info["fsc"]:
                policy_info = {"mem_node": item.policy_info["mem_node"]}
            else:
                policy_info = item.policy_info["rl"]
            modified_item = Trajectory(
                step_type=item.step_type,
                observation=item.observation["observation"],
                action=item.action,
                policy_info=(policy_info),
                next_step_type=item.next_step_type,
                reward=item.reward,
                discount=item.discount,
            )
            if item.policy_info["fsc"]:
                replay_buffer_fsc._add_batch(modified_item)
            else:
                replay_buffer_rl._add_batch(modified_item)

        return _add_batch
