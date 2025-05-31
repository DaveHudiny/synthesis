# Description: This file contains the implementation of the FatherAgent class, which is the parent class of other agents in the project.
# Author: David Hudák
# Login: xhudak03
# Project: diploma-thesis
# File: father_agent.py

from tf_agents.policies import py_tf_eager_policy


from environment import tf_py_environment

from environment.sparse_reward_shaper import SparseRewardShaper, RewardShaperMethods, ObservationLevel

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.trajectories import Trajectory


import tensorflow as tf
import tf_agents

from tools.trajectory_buffer import TrajectoryBuffer
from tools.encoding_methods import *
from agents.abstract_agent import AbstractAgent
from agents.random_agent import RandomAgent
from tools.evaluation_results_class import EvaluationResults
from tools.evaluators import *
from agents.policies.fsc_copy import FSC
from agents.policies.combination_policy import CombinationPolicy, CombinationSettings
from tools.args_emulator import ArgsEmulator, ReplayBufferOptions

from agents.policies.policy_mask_wrapper import PolicyMaskWrapper
from agents.policies.simple_fsc_policy import *
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

from agents.reward_driver import DynamicRewardDriver

from reward_machines.predicate_automata import PredicateAutomata, create_dummy_predicate_automata

import logging

logger = logging.getLogger(__name__)

# max_num_steps * MULTIPLIER = maximum length of replay buffer for each thread.
OFF_POLICY_BUFFER_SIZE_MULTIPLIER = 500

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse_environment(environment, potential_dict_key="train_model"):
    if isinstance(environment, dict):
        return environment[potential_dict_key]
    else:
        return environment


class AgentSettings:
    """Class for storing information about agents. Possible usage with extension of the project."""

    def __init__(self, preprocessing_layers=[150, 150], lstm_units=[100], postprocessing_layers=[]):
        self.preprocessing_layers = preprocessing_layers
        self.lstm_units = lstm_units
        self.postprocessing_layers = postprocessing_layers


class FatherAgent(AbstractAgent):
    """Class for the parent agent of all agents in the project."""

    def load_fsc(self, fsc_json_path):
        """Load FSC from JSON file.

        Args:
            fsc_json_path: Path to the JSON file with FSC.
        """
        with open(fsc_json_path, 'r') as f:
            fsc_json = json.load(f)
        fsc = FSC.from_json(fsc_json)
        return fsc

    def common_init(self, environment: EnvironmentWrapperVec, tf_environment: tf_py_environment.TFPyEnvironment,
                    args: ArgsEmulator, load=False, agent_folder=None, wrapper: tf_agents.policies.tf_policy.TFPolicy = None):
        """Common initialization of the agents.

        Args:
            environment: The environment wrapper object, used for additional information about the environment.
            tf_environment: The TensorFlow environment object, used for simulation information.
            args: The arguments object for all the important settings.
            load: Whether to load the agent. Unused.
            agent_folder: The folder where the agent is stored.
        """
        self.environment = environment
        self.tf_environment = tf_environment
        self.args = args
        self.evaluation_episodes = args.evaluation_episodes
        self.agent_folder = agent_folder
        self.traj_num_steps = args.trajectory_num_steps
        self.agent = None
        self.observation_and_action_constraint_splitter = observation_and_action_constraint_splitter
        if args.paynt_fsc_imitation:
            self.fsc = self.load_fsc(args.paynt_fsc_json)
        self.wrapper = wrapper
        self.wrapper_eager = None
        self.evaluation_result = EvaluationResults(self.environment.goal_value)
        self.duplexing = False

    def __init__(self, environment: EnvironmentWrapperVec, tf_environment: tf_py_environment.TFPyEnvironment, args, load=False, agent_folder=None):
        """Initialization of the father agent. Not recommended to use this class directly, use the child classes instead. Implemented as example.

        Args:
            environment: The environment wrapper object, used for additional information about the environment.
            tf_environment: The TensorFlow environment object, used for simulation information.
            args: The arguments object for all the important settings.
            load: Whether to load the agent. Unused.
            agent_folder: The folder where the agent is stored."""

        self.common_init(environment, tf_environment,
                         args, load, agent_folder)
        # Initialize random policy agent
        self.agent = RandomAgent(tf_environment.time_step_spec(),
                                 tf_environment.action_spec())
        self.init_replay_buffer()
        self.init_collector_driver(tf_environment)
        self.init_vec_evaluation_driver(
            self.tf_environment, self.environment, self.args.max_steps)

    def init_replay_buffer(self, buffer_size=None):
        """Initialize the uniform replay buffer for the agent.

        Args:
            tf_environment: The TensorFlow environment object, used for providing important specifications.
        """
        if buffer_size is None:
            buffer_size = self.args.buffer_size
        if self.args.replay_buffer_option == ReplayBufferOptions.ORIGINAL_OFF_POLICY or not self.args.vectorized_envs_flag:
            batch_size = 1
        else:
            batch_size = self.args.num_environments
            if self.args.replay_buffer_option == ReplayBufferOptions.OFF_POLICY:
                buffer_size = self.args.max_steps * OFF_POLICY_BUFFER_SIZE_MULTIPLIER
            elif self.args.replay_buffer_option == ReplayBufferOptions.ON_POLICY:
                buffer_size = self.args.trajectory_num_steps + 10

        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=batch_size,
            max_length=buffer_size)

    def init_vec_evaluation_driver(self, tf_environment: tf_py_environment.TFPyEnvironment, environment: EnvironmentWrapperVec, num_steps=400):
        """Initialize the vectorized evaluation driver for the agent. Used for evaluation of the agent.

        Args:
            environment: The vectorized environment object, used for simulation information.
            num_steps: The number of steps for evaluation.
        """
        self.trajectory_buffer = TrajectoryBuffer(environment)
        eager = py_tf_eager_policy.PyTFEagerPolicy(
            self.get_evaluation_policy(), use_tf_function=True, batch_time_steps=False)
        observers = [self.trajectory_buffer.add_batched_step]
        if self.args.curiosity_automata_reward:
            self.vec_driver = DynamicRewardDriver(
                tf_environment,
                eager,
                observers=observers,
                num_steps=(1 + num_steps) * self.environment.num_envs,
                trajectory_reward_generator=self.collect_policy_wrapper.generate_curiosity_reward)
        else:
            self.vec_driver = DynamicStepDriver(
                tf_environment,
                eager,
                observers=observers,
                num_steps=(1 + num_steps) * self.environment.num_envs
            )

    def get_observers(self, alternative_observer):
        if alternative_observer is None:
            observers = [self.replay_buffer.add_batch]
        else:
            observers = [alternative_observer]
        return observers

    def compute_number_of_steps(self):
        if not self.args.vectorized_envs_flag:
            num_steps = self.args.trajectory_num_steps
        elif self.args.replay_buffer_option == ReplayBufferOptions.ON_POLICY:
            # TODO: Compare it with self.args.max_steps
            num_steps = self.args.num_environments * self.args.trajectory_num_steps
        elif self.args.replay_buffer_option == ReplayBufferOptions.OFF_POLICY:
            num_steps = self.args.num_environments * \
                (self.args.trajectory_num_steps // 2)
        elif self.args.replay_buffer_option == ReplayBufferOptions.ORIGINAL_OFF_POLICY:
            num_steps = self.args.trajectory_num_steps
        else:
            num_steps = self.args.trajectory_num_steps
        return num_steps

    class BatchEnvironmentDriver:
        def __init__(self,
                     batch_tf_environments: list[tf_py_environment.TFPyEnvironment],
                     eager: py_tf_eager_policy.PyTFEagerPolicy,
                     observers: list[callable],
                     num_steps):
            self.batch_tf_environments = batch_tf_environments
            self.eager = eager
            self.observers = observers
            self.num_steps = num_steps
            self.current_env = 0
            self.drivers = self.__generate_each_driver()

        def __generate_each_driver(self) -> list[DynamicStepDriver]:
            drivers = []
            for env in self.batch_tf_environments:
                driver = DynamicStepDriver(
                    env,
                    self.eager,
                    observers=self.observers,
                    num_steps=self.num_steps
                )
                drivers.append(driver)

            return drivers

        def run(self):
            if self.current_env >= 500:
                self.drivers[-1].run()
            else:
                self.drivers[self.current_env % len(self.drivers)].run()
                self.current_env = self.current_env + 1

    def create_go_explore_observer(self, environment: EnvironmentWrapperVec):
        def _add_batch(item: Trajectory):
            environment.go_explore_add_state(
                item.policy_info["current_automata_state"].numpy())
        return _add_batch
    
    def init_collector_driver(self, tf_environment: tf_py_environment.TFPyEnvironment,
                              demasked: bool = False,
                              alternative_observer: callable = None,
                              batch_tf_environments: list[tf_py_environment.TFPyEnvironment] = []):
        if self.args.curiosity_automata_reward or self.args.go_explore or self.args.predicate_automata_obs:
            predicate_automata = self.environment.predicate_automata
        else:
            predicate_automata = None
        if demasked:
            self.collect_policy_wrapper = PolicyMaskWrapper(
                self.agent.collect_policy, observation_and_action_constraint_splitter, tf_environment.time_step_spec(), predicate_automata=predicate_automata)
            if self.args.masked_training:
                self.collect_policy_wrapper.set_policy_masker()
            else:
                self.collect_policy_wrapper.unset_policy_masker()
            eager = py_tf_eager_policy.PyTFEagerPolicy(
                self.collect_policy_wrapper, use_tf_function=True, batch_time_steps=False)
        else:
            eager = py_tf_eager_policy.PyTFEagerPolicy(
                self.agent.collect_policy, use_tf_function=True, batch_time_steps=False)
        if demasked:
            observers = [self.get_demasked_observer(
                self.args.vectorized_envs_flag)]
        else:
            observers = self.get_observers(alternative_observer)
        if self.args.go_explore:
            self.environment.set_go_explore(predicate_automata)
        num_steps = self.compute_number_of_steps()
        self.num_steps = num_steps
        if len(batch_tf_environments) > 0:
            # Add original tf_environment as last to enable training on it.
            batch_tf_environments.append(tf_environment)
            self.driver = self.BatchEnvironmentDriver(
                batch_tf_environments, eager, observers, num_steps)
        elif self.args.curiosity_automata_reward:
            self.driver = DynamicRewardDriver(
                tf_environment,
                eager,
                observers=observers,
                num_steps=num_steps,
                trajectory_reward_generator=self.collect_policy_wrapper.generate_curiosity_reward)
        else:
            self.driver = DynamicStepDriver(
                tf_environment,
                eager,
                observers=observers,
                num_steps=num_steps)

    def init_random_collector_driver(self, tf_environment: tf_py_environment.TFPyEnvironment,
                                     alternative_observer: callable = None):
        """Initialize the random policy collector driver for the agent. Used for random exploration.

        Args:
            tf_environment: The TensorFlow environment object, used for simulation information.
        """
        random_policy = tf_agents.policies.random_tf_policy.RandomTFPolicy(tf_environment.time_step_spec(),
                                                                           tf_environment.action_spec(),
                                                                           observation_and_action_constraint_splitter=self.observation_and_action_constraint_splitter)
        observers = self.get_observers(alternative_observer)
        if not hasattr(self, "num_steps"):
            self.num_steps = self.args.trajectory_num_steps
        self.random_driver = DynamicStepDriver(
            tf_environment,
            random_policy,
            observers=observers,
            num_steps=self.num_steps
        )

    def get_initial_state(self, batch_size=None):
        """Get the initial state of the agent."""
        return self.agent.policy.get_initial_state(batch_size=batch_size)

    def init_fsc_policy_driver(self, tf_environment: tf_py_environment.TFPyEnvironment, fsc: FSC = None):
        """Initializes the driver for the FSC policy. Used for hard and soft FSC advices. Currently implemented only for wrapped policies."""
        parallel_policy = self.wrapper
        self.fsc_policy = SimpleFSCPolicy(fsc, self.environment.action_keywords, self.tf_environment.time_step_spec(), self.tf_environment.action_spec(),
                                          policy_state_spec=(), info_spec=(), observation_and_action_constraint_splitter=fsc_action_constraint_splitter)
        self.fsc_policy = py_tf_eager_policy.PyTFEagerPolicy(
            self.fsc_policy, use_tf_function=True, batch_time_steps=False)
        self.combination_policy = CombinationPolicy(policies=[self.fsc_policy, parallel_policy], time_step_spec=self.tf_environment.time_step_spec(),
                                                    action_spec=self.tf_environment.action_spec(), observation_and_action_constraint_splitter=self.observation_and_action_constraint_splitter,
                                                    enable_masking=False, combination_settings=CombinationSettings.PRIMARY_POLICY, primary_policy_index=0)
        self.combination_policy = py_tf_eager_policy.PyTFEagerPolicy(
            self.combination_policy, use_tf_function=True, batch_time_steps=False)
        observer = self.get_demasked_observer()
        self.fsc_driver = DynamicStepDriver(
            tf_environment,
            self.combination_policy,
            observers=[observer],
            num_steps=self.args.num_environments * self.args.trajectory_num_steps
        )

    def get_evaluation_policy(self):
        """Get the policy for evaluation. Important, when using wrappers."""
        if self.wrapper is None:
            return self.agent.policy
        else:
            self.wrapper.set_policy_masker()
            return self.wrapper

    def train_innerest_body(self, experience, train_iteration, randomized=False, vectorized=False):
        """Train the agent with the experience. Used for training the agent with the experience.

        Args:
            experience: The experience for training the agent.
        """
        train_loss = self.agent.train(experience).loss
        train_loss = train_loss.numpy()
        self.agent.train_step_counter.assign_add(1)
        self.evaluation_result.add_loss(train_loss)
        if train_iteration % 5 == 0:
            logger.info(
                f"Step: {train_iteration}, Training loss: {train_loss}")
        if train_iteration % 100 == 0:
            self.environment.set_random_starts_simulation(False)
            self.evaluate_agent(vectorized=vectorized,
                                max_steps=self.args.max_steps * 2)
            
            self.environment.set_random_starts_simulation(randomized)
            self.tf_environment.reset()

        return train_loss

    def train_body_off_policy(self, num_iterations, vectorized: bool = True, randomized=False):
        """Train the agent off-policy via original method (1 x num_steps). Main training function for the agents.

        Args:
            num_iterations: The number of iterations for training.
            vectorized: Whether to use vectorized environment for training.
            randomized: Whether to use random starts for simulation.
        """

        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=8, sample_batch_size=self.args.batch_size, num_steps=self.traj_num_steps, single_deterministic_pass=False).prefetch(8)
        iterator = iter(self.dataset)
        logger.info("Training agent off-policy original")
        self.environment.set_random_starts_simulation(
            randomized_bool=randomized)
        self.tf_environment.reset()

        for _ in range(self.args.trajectory_num_steps):
            self.driver.run()

        for i in range(num_iterations):
            self.driver.run()
            experience, _ = next(iterator)
            self.train_innerest_body(
                experience, i, randomized=randomized, vectorized=vectorized)

    def train_body_on_policy(self, num_iterations, vectorized: bool = True, randomized=False):
        """Train the agent on-policy. Main training function for the agents.

        Args:
            num_iterations: The number of iterations for training.
            vectorized: Whether to use vectorized environment for training.
            randomized: Whether to use random starts for simulation.
        """
        self.replay_buffer.clear()
        self.environment.set_random_starts_simulation(
            randomized_bool=randomized)
        self.tf_environment.reset()
        logger.info("Training agent on-policy")
        self.flag_restarted = True
        for i in range(num_iterations):
            if not self.jumpstarting and self.fsc_training and i <= num_iterations // 4:
                self.run_special_runner()
            elif self.jumpstarting and i % 7 == 0 and i < num_iterations // 4:
                self.perform_jumpstart(self.switch_probability)
                self.driver.run()
            else:
                self.driver.run()
            if i == num_iterations // 4:
                self.environment.unset_reward_shaper()
            data = self.replay_buffer.gather_all()
            self.train_innerest_body(
                data, i, randomized=randomized, vectorized=vectorized)
            self.replay_buffer.clear()

    def run_special_runner(self):
        """Run the runner for the agent. Used for running the agent."""
        if self.fsc_training and self.switch_probability is None:
            self.fsc_driver.run()
        elif self.fsc_training and self.switch_probability is not None:
            self.fsc_driver.run()
            if np.random.rand() < self.switch_probability / 10:
                self.fsc_policy.reset_switching()
                self.tf_environment.reset()

    def init_demonstration_shaper(self, fsc: FSC):
        self.shaper = SparseRewardShaper(RewardShaperMethods.DEMONSTRATION, ObservationLevel.STATE_ACTION, maximum_reward=1.9,
                                         batch_size=self.args.batch_size, buffer_length=101, cyclic_buffer=True, observation_length=1, action_length=1)

        def _add_batch(item: Trajectory):
            observation = item.observation["integer"]
            states = self.environment.prev_states
            self.shaper.add_demonstration(

                item.action, observation=observation, states=states)

        fsc_policy = SimpleFSCPolicy(fsc, self.environment.action_keywords, self.tf_environment.time_step_spec(), self.tf_environment.action_spec(),
                                     policy_state_spec=(), info_spec=(), observation_and_action_constraint_splitter=fsc_action_constraint_splitter)

        fsc_policy = py_tf_eager_policy.PyTFEagerPolicy(
            fsc_policy, use_tf_function=True, batch_time_steps=False)
        demonstration_driver = DynamicStepDriver(
            self.tf_environment, fsc_policy, observers=[_add_batch], num_steps=self.args.batch_size * 100)
        demonstration_driver.run()

    def init_reward_shaping(self, fsc: FSC):
        self.init_demonstration_shaper(fsc)
        self.environment.set_reward_shaper(
            self.shaper.create_reward_function())

    def special_agent_pretraining_stuff(self):
        pass  # Father agent does not have any special pretraining stuff.

    def special_agent_midtraining_stuff(self):
        pass  # Father agent does not have any special training stuff.

    def train_agent(self, iterations: int,
                    vectorized: bool = True,
                    replay_buffer_option: ReplayBufferOptions = ReplayBufferOptions.ON_POLICY,
                    fsc: FSC = None,
                    jumpstart_fsc: bool = False,
                    debug: bool = False,
                    shaping=False):
        """Trains agent with the principle of using gather all on replay buffer and clearing it after each iteration.

        Args:
            iterations (int): Number of iterations to train agent.
        """
        if not debug:
            self.agent.train = common.function(self.agent.train)
        if fsc is not None:
            self.evaluate_fsc(fsc)
        if fsc is not None and shaping:
            self.init_reward_shaping(fsc)
        if fsc is not None and not shaping:
            if jumpstart_fsc:
                self.switch_probability = 0.05
                self.jumpstarting = True
                self.init_jumpstarting(fsc)
            else:
                self.switch_probability = None
                self.jumpstarting = False
                self.init_fsc_policy_driver(
                    self.tf_environment, fsc)

            self.fsc_training = True
        else:
            self.fsc_training = False
            self.jumpstarting = False

        logger.info("Training agent with replay buffer option: {0}".format(
            replay_buffer_option))
        logger.info("Before training evaluation.")
        self.evaluate_agent(vectorized=vectorized,
                            max_steps=self.args.max_steps * 2)
        if replay_buffer_option == ReplayBufferOptions.ORIGINAL_OFF_POLICY or replay_buffer_option == ReplayBufferOptions.OFF_POLICY:
            self.train_body_off_policy(iterations, vectorized, randomized=self.args.random_start_simulator)
        if replay_buffer_option == ReplayBufferOptions.ON_POLICY:
            self.train_body_on_policy(iterations, vectorized, randomized=self.args.random_start_simulator)
        logger.info("Training finished.")
        self.environment.set_random_starts_simulation(False)
        self.evaluate_agent(vectorized=vectorized, last=True,
                            max_steps=self.args.max_steps * 2)

    def get_throw_away_driver(self, fsc: FSC):

        fsc_policy = SimpleFSCPolicy(fsc, self.environment.action_keywords, self.tf_environment.time_step_spec(), self.tf_environment.action_spec(),
                                     policy_state_spec=(), info_spec=(), observation_and_action_constraint_splitter=fsc_action_constraint_splitter)

        eager = py_tf_eager_policy.PyTFEagerPolicy(
            fsc_policy, use_tf_function=True, batch_time_steps=False)
        num_of_fsc_steps = np.random.geometric(0.05)
        throw_away_fsc_driver = DynamicStepDriver(
            self.tf_environment, eager, observers=[], num_steps=self.args.num_environments * num_of_fsc_steps)
        return throw_away_fsc_driver

    def init_jumpstarting(self, fsc: FSC, saynt: bool = False):
        if saynt:
            raise NotImplementedError(
                "SAYNT jumpstarting is not implemented yet.")
        self.throw_away_driver = self.get_throw_away_driver(fsc)

    def perform_jumpstart(self, geometric_distribution_parameter: float = 0.05):
        self.tf_environment.reset()
        num_of_fsc_steps = np.random.geometric(
            geometric_distribution_parameter)
        for _ in range(num_of_fsc_steps):
            self.throw_away_driver.run()

    def is_rl_better(self, evaluation_result: EvaluationResults, performance_condition: float):
        import math

        if performance_condition is None or math.isnan(performance_condition):
            return True
        # TODO: Find better way, how to check conditions.
        if performance_condition > 1.001:
            if evaluation_result.best_return <= performance_condition * 1.25:
                return True
            return False
        else:  # Reachability condition
            if evaluation_result.best_reach_prob >= performance_condition * 0.75:
                return True
            return False

    def render_agent_behavior(self, policy: TFPolicy):
        self.environment.set_random_starts_simulation(False)
        self.set_agent_greedy()
        self.tf_environment.reset()
        policy_state = policy.get_initial_state(self.tf_environment.batch_size)
        trajectory = [self.environment.render("rgb_array")]
        while True:
            time_step = self.tf_environment.current_time_step()
            action = policy.action(time_step, policy_state=policy_state)
            policy_state = action.state
            self.tf_environment.step(action.action)
            trajectory.append(self.environment.render("rgb_array"))
            if self.environment.dones[0]:
                break
        self.environment.render("human", trajectory=trajectory)
        self.tf_environment.reset()

    def set_policy_masking(self):
        """If PPO, this function sets the masking active for agent wrapper."""
        pass

    def unset_policy_masking(self):
        """If PPO, this function sets the masking inactive for agent wrapper."""
        pass

    def evaluate_agent(self, last=False, vectorized=False, max_steps: int = None):
        """Evaluate the agent. Used for evaluation of the agent during training.

        Args:
            last: Whether this is the last evaluation of the agent.
        """
        self.environment.set_random_starts_simulation(False)
        self.environment.temporarily_set_num_envs(512)
        if not hasattr(self, "tf_env_eval"):
            self.tf_env_eval = tf_py_environment.TFPyEnvironment(
                self.environment)
        if self.args.go_explore:
            self.environment.unset_go_explore()
        if self.args.prefer_stochastic:
            self.set_agent_stochastic()
        else:
            self.set_agent_greedy()
            self.set_policy_masking()
        if not vectorized:
            if last:
                evaluation_episodes = self.evaluation_episodes * 2
            else:
                evaluation_episodes = self.evaluation_episodes
            compute_average_return(
                self.get_evaluation_policy(), self.tf_environment, evaluation_episodes, self.environment, self.evaluation_result.update)
        else:
            if not hasattr(self, "vec_driver"):
                self.init_vec_evaluation_driver(
                    self.tf_env_eval, self.environment, num_steps=self.args.max_steps + 5)
            if self.args.replay_buffer_option == ReplayBufferOptions.ORIGINAL_OFF_POLICY:
                self.environment.set_num_envs(
                    self.args.batch_size)
            self.tf_env_eval.reset()
            if last:
                self.set_agent_greedy()

                logger.info("Evaluating agent with greedy masked policy.")
                self.set_policy_masking()
                if self.args.render_if_possible and self.environment.grid_like_renderer:
                    self.render_agent_behavior(self.get_evaluation_policy())
            self.vec_driver.run()
            if self.args.replay_buffer_option == ReplayBufferOptions.ORIGINAL_OFF_POLICY:
                self.environment.set_num_envs(1)
                self.tf_env_eval.reset()
            self.trajectory_buffer.final_update_of_results(
                self.evaluation_result.update)
            self.trajectory_buffer.clear()

        self.set_agent_stochastic()
        self.unset_policy_masking()
        if self.evaluation_result.best_updated and self.agent_folder is not None:
            self.save_agent(best=True)
        self.log_evaluation_info()
        if self.args.go_explore:
            self.environment.set_go_explore()
        self.environment.reset_num_envs()

    def log_evaluation_info(self):
        logger.info('Average Return = {0}'.format(
            self.evaluation_result.returns[-1]))
        logger.info('Average Virtual Goal Value = {0}'.format(
            self.evaluation_result.returns_episodic[-1]))
        logger.info('Goal Reach Probability = {0}'.format(
            self.evaluation_result.reach_probs[-1]))
        logger.info('Trap Reach Probability = {0}'.format(
            self.evaluation_result.trap_reach_probs[-1]))
        logger.info('Variance of Return = {0}'.format(
            self.evaluation_result.each_episode_variance[-1]))
        logger.info('Current Best Return = {0}'.format(
            self.evaluation_result.best_return))
        logger.info('Current Best Reach Probability = {0}'.format(
            self.evaluation_result.best_reach_prob))

    def set_agent_greedy(self):
        """Set the agent for to be greedy for evaluation. Used only with PPO agent, where we select greedy evaluation.
        """
        if self.wrapper is None:
            pass
        else:
            self.wrapper.set_greedy(True)

    def set_agent_stochastic(self):
        """Set the agent to be stochastic for evaluation. Used only with PPO agent, where we select stochastic evaluation.
        """
        if self.wrapper is None:
            pass
        else:
            self.wrapper.set_greedy(False)

    def action(self, time_step, policy_state=None):
        """Make a decision based on the policy of the agent."""
        if policy_state is None:
            policy_state = self.wrapper.get_initial_state(self.tf_environment.batch_size)
        if self.wrapper_eager is not None:
            return self.wrapper_eager.action(
                time_step, policy_state=policy_state)
        return self.agent.policy.action(time_step, policy_state=policy_state)
    
    def get_policy(self, eager=True):
        """Get the policy of the agent."""
        if self.wrapper is not None:
            if eager:
                return PyTFEagerPolicy(self.wrapper, use_tf_function=True, batch_time_steps=False)
            else:
                return self.wrapper
        if eager:
            return PyTFEagerPolicy(self.agent.policy, use_tf_function=True, batch_time_steps=False)
        else:
            return self.agent.policy

    def save_agent(self, best=False):
        """Save the agent. Used for saving the agent after or during training training.

        Args:
            best: Whether this is the best agent. If true, the agent is saved in the best folder.
        """
        if self.agent is None or tf.train is None:
            logger.info("No agent for saving.")
            return
        checkpoint = tf.train.Checkpoint(agent=self.agent)
        if best:
            agent_folder = self.agent_folder + "/best"
        else:
            agent_folder = self.agent_folder
        manager = tf.train.CheckpointManager(
            checkpoint, agent_folder, max_to_keep=5)
        manager.save()

    def load_agent(self, best=False):
        """Load the agent. Used for loading the agent after or during training.

        Args:
            best: Whether this is the best agent. If true, the agent is loaded from the best folder.
        """
        checkpoint = tf.train.Checkpoint(agent=self.agent)
        if best:
            agent_folder = self.agent_folder + "/best"
        else:
            agent_folder = self.agent_folder
        manager = tf.train.CheckpointManager(
            checkpoint, agent_folder, max_to_keep=5)
        latest_checkpoint = manager.latest_checkpoint
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            logger.info(f"Loaded data from checkpoint: {latest_checkpoint}")
        else:
            logger.info("No data for loading.")

    def reset_weights(self):
        """Reset the weights of the agent. Implemented in the child classes."""
        raise NotImplementedError

    def get_demasked_observer(self, vectorized=False):
        """Observer for replay buffer. Used to demask the observation in the trajectory. Used with policy wrapper."""
        def _add_batch(item: Trajectory):
            # item.policy_info["dist_params"]["logits"] = item.policy_info["dist_params"]["logits"]
            modified_item = Trajectory(
                step_type=item.step_type,
                observation=item.observation["observation"],
                action=item.action,
                policy_info={"dist_params": item.policy_info["dist_params"]},
                next_step_type=item.next_step_type,
                reward=item.reward,
                discount=item.discount,
            )
            self.replay_buffer._add_batch(modified_item)
        return _add_batch

    def get_action_handicapped_observer(self):
        def _add_batch(item: Trajectory):
            modified_item = Trajectory(
                step_type=item.step_type,
                observation=item.observation["observation"],
                action=tf.constant(0, tf.int32),
                policy_info=(item.policy_info),
                next_step_type=item.next_step_type,
                reward=item.reward,
                discount=item.discount,
            )
            self.replay_buffer._add_batch(modified_item)
        return _add_batch

    def get_jumpstart_observer(self):
        def _add_batch(item: Trajectory):
            if self.fsc_policy.switched:
                self.replay_buffer._add_batch(item)
        return _add_batch

    def evaluate_fsc(self, fsc: FSC):
        fsc_policy = SimpleFSCPolicy(fsc, self.environment.action_keywords, self.tf_environment.time_step_spec(), self.tf_environment.action_spec(),
                                     policy_state_spec=(), info_spec=(), observation_and_action_constraint_splitter=fsc_action_constraint_splitter)
        eager = py_tf_eager_policy.PyTFEagerPolicy(
            fsc_policy, use_tf_function=True, batch_time_steps=False)
        trajectory_buffer = TrajectoryBuffer(self.environment)
        vec_driver = DynamicStepDriver(
            self.tf_environment, eager, observers=[trajectory_buffer.add_batched_step], num_steps=self.args.num_environments * (self.args.max_steps + 1))
        vec_driver.run()

        def print_results(avg_return, avg_episode_return, reach_prob, episode_variance, num_episodes,
                          trap_reach_prob, virtual_variance, combined_variance):
            logger.info("Average return: {0}".format(avg_return))
            logger.info("Average episode return: {0}".format(
                avg_episode_return))
            logger.info("Reachability probability: {0}".format(reach_prob))
            logger.info("Episode variance: {0}".format(episode_variance))
            logger.info("Number of episodes: {0}".format(num_episodes))
            logger.info(
                "Trap reachability probability: {0}".format(trap_reach_prob))
            logger.info("Virtual variance: {0}".format(virtual_variance))
            logger.info("Combined variance: {0}".format(combined_variance))

        trajectory_buffer.final_update_of_results(print_results)
