# This file contains the Synthesizer_RL class, which creates interface between RL and PAYNT.
# Author: David Hudák
# Login: xhudak03
# File: synthesizer_rl.py


from rl_src.environment.environment_wrapper import Environment_Wrapper
from rl_src.experimental_interface import ArgsEmulator, ExperimentInterface
from rl_src.interpreters.tracing_interpret import TracingInterpret
from rl_src.agents.policies.parallel_fsc_policy import FSC_Policy
from rl_src.tools.saving_tools import save_statistics_to_new_json
from rl_src.tools.encoding_methods import *
from rl_src.tools.evaluators import EvaluationResults
from paynt.quotient.fsc import FSC
from rl_src.agents.recurrent_ppo_agent import Recurrent_PPO_agent

from rl_src.environment import tf_py_environment

from paynt.quotient.fsc import FSC
from paynt.rl_extension.saynt_controller.saynt_driver import SAYNT_Driver

from rl_src.agents.duplexing.behavioral_trainers import Actor_Value_Pretrainer


import logging

logger = logging.getLogger(__name__)


class AgentsWrapper:
    """Class for the interface between RL and PAYNT.
    """

    def __init__(self, stormpy_model, args: ArgsEmulator,
                 initial_fsc_multiplier: float = 1.0,
                 qvalues: list = None, action_labels_at_observation: dict = None,
                 random_init_starts_q_vals: bool = False,
                 pretrain_dqn = False):
        """Initialization of the interface.
        Args:
            stormpy_model: Model of the environment.
            args (ArgsEmulator): Arguments for the initialization.
            initial_fsc_multiplier (float, optional): Initial soft FSC multiplier. Defaults to 1.0.
        """

        self.interface = ExperimentInterface(args, stormpy_model)
        self.random_initi_starts_q_vals = random_init_starts_q_vals

        self.interface.environment, self.interface.tf_environment = self.interface.initialize_environment(self.interface.args, self.interface.pomdp_model)
        logger.info("RL Environment initialized")
        self.agent = self.interface.initialize_agent(
            qvalues_table=qvalues, action_labels_at_observation=action_labels_at_observation)
        
        self.interpret = TracingInterpret(self.interface.environment, self.interface.tf_environment,
                                          self.interface.args.encoding_method)
        self.fsc_multiplier = initial_fsc_multiplier

    def train_agent(self, iterations: int):
        """Train the agent.
        Args:
            iterations (int): Number of iterations.
        """
        self.agent.train_agent(iterations)
        self.agent.save_agent()

    def train_agent_qval_randomization(self, iterations: int, qvalues: list):
        self.interface.environment.set_new_qvalues_table(
            qvalues_table=qvalues
        )
        self.agent.train_agent_off_policy(iterations, q_vals_rand=True)

    def interpret_agent(self, best: bool = False, with_refusing: bool = False, greedy: bool = False, randomize_illegal_actions: bool = True):
        """Interpret the agent.
        Args:
            best (bool, optional): Whether to use the best, or the last trained agent. Defaults to False.
            with_refusing (bool, optional): Whether to use refusing. Defaults to False.
            greedy (bool, optional): Whether to use greedy policy evaluation (in case of PPO). Defaults to False.
        Returns:
            dict: Dictionary of the interpretation.
        """
        if best:
            self.agent.load_agent(best)
        # Works only with agents which use policy wrapping (in our case only PPO)
        if greedy:
            self.agent.set_agent_greedy()
        else:
            self.agent.set_agent_stochastic()
        result = self.interpret.get_dictionary(self.agent, with_refusing, randomize_illegal_actions=randomize_illegal_actions, vectorized=True)
        self.agent.set_agent_stochastic()
        return result

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
            self.interface.tf_environment, fsc=fsc)
        self.agent.train_agent_off_policy(iterations)

    def sample_trajectories_with_fsc(self, episodes = 10, fsc = None, soft_decision = False, fsc_multiplier = None, switch_probability = False):
        from rl_src.tools.encoding_methods import observation_and_action_constraint_splitter
        fsc_policy = FSC_Policy(self.interface.tf_environment, fsc,
                                     observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
                                     tf_action_keywords=self.interface.environment.action_keywords,
                                     info_spec=(),
                                     soft_decision=soft_decision,
                                     soft_decision_multiplier=fsc_multiplier,
                                     switch_probability=switch_probability)

        tf_environment = self.interface.tf_environment
        environment = self.interface.environment
        environment.set_random_starts_simulation(False)
        from paynt.rl_extension.saynt_rl_tools.trajectory_collector import collect_trajectories
        episodes = collect_trajectories(num_of_episodes=10, policy=fsc_policy, environment=environment, tf_environment=tf_environment)
        
        # Print statistics
        for episode in episodes:
            print(len(episode))
            print(episode[-1].success)

        return episodes
        
    def train_agent_combined_with_fsc_advanced(self, iterations : int = 1000, fsc: FSC = None, condition : float = None):
        """_summary_

        Args:
            iterations (int, optional): _description_. Defaults to 1000.
            fsc (FSC, optional): _description_. Defaults to None.
            condition (float, optional): _description_. Defaults to None.
        """
        self.agent.train_agent(iterations,
                               vectorized=self.interface.args.vectorized_envs_flag, 
                               replay_buffer_option=self.interface.args.replay_buffer_option,
                               fsc=fsc, jumpstart_fsc=False)
        return
        self.agent.mixed_fsc_train(iterations, on_policy=False, performance_condition=condition, fsc=fsc, soft_fsc=False, switch_probability = 0.05)

    def save_to_json(self, experiment_name: str = "PAYNTc+RL", model="model", method="PPO", time: float = -1.0):
        """Save the agent to JSON.
        Args:
            experiment_name (str): Name of the experiment.
        """
        evaluation_result = self.agent.evaluation_result
        save_statistics_to_new_json(
            experiment_name, model, method, evaluation_result, self.interface.args, time)

    def get_encoding_method(self, method_str: str = "Valuations") -> EncodingMethods:
        if method_str == "Valuations":
            return EncodingMethods.VALUATIONS
        elif method_str == "One-Hot":
            return EncodingMethods.ONE_HOT_ENCODING
        elif method_str == "Valuations++":
            return EncodingMethods.VALUATIONS_PLUS
        else:
            return EncodingMethods.INTEGER

    def get_saynt_trajectories(self, storm_control, quotient, fsc: FSC = None, q_values=None, model_reward_multiplier=-1.0):
        
        args = self.interface.get_args()
        pre_trainer = Actor_Value_Pretrainer(self.interface.environment, self.interface.tf_environment,
                                             args, self.agent.agent.collect_data_spec)
        agent_folder = f"./trained_agents/{args.agent_name}_{args.learning_method}_{args.encoding_method}"
        actor = pre_trainer.actor_net
        critic = pre_trainer.critic_net
        self.agent = Recurrent_PPO_agent(self.interface.environment, self.interface.tf_environment, 
                        args, args.load_agent, agent_folder, actor_net=actor, critic_net=critic)
        observer = pre_trainer.replay_buffer._add_batch
        tf_action_labels = self.interface.environment.action_keywords
        if not hasattr(self, "saynt_driver"):
            print("Creating new SAYNT driver")
            encoding_method = self.get_encoding_method(
                self.interface.args.encoding_method)
            self.saynt_driver = SAYNT_Driver([observer], storm_control, quotient,
                                             tf_action_labels, encoding_method, self.interface.args.discount_factor,
                                             fsc=fsc, q_values=q_values, model_reward_multiplier=model_reward_multiplier)
        # self.agent.train_agent_off_policy(101, random_init=True)
        self.saynt_driver.episodic_run(300)
        for i in range(10):
            self.saynt_driver.episodic_run(30)
            pre_trainer.train_both_networks(200, fsc=fsc, use_best_traj_only=False, offline_data=True)
        # TODO: self.agent.train_agent_off_policy(2000, random_init=False, probab_random_init_state=0.1)
            


    def check_four_phase_condition(self, fsc_quality : float, maximizing_value : bool, probability_cond : bool, evaluation_result : EvaluationResults):
        if probability_cond:
            best_value = evaluation_result.best_reach_prob 
        else:
            best_value = evaluation_result.best_return
            if not maximizing_value:
                best_value = -best_value # Minimizing reward RL works with negative rewards.
        if maximizing_value:
            return fsc_quality > best_value
        else:
            return fsc_quality < best_value

    # "only_pretrained", "only_duplex", "only_duplex_critic", "complete", "four_phase"
    # fsc_quality is either minimized or maximized. Condition given quality of FSC is currently used only in the four_phase implementation.
    # Condition can optimize probability (e.g. maximize probability of reaching the goal state) or reward (e.g. minimize number of steps to reach the goal state)
    def train_with_bc(self, fsc : FSC = None, sub_method = "only_pretrained", nr_of_iterations : int = 1000):
        # self.dqn_agent.pre_train_with_fsc(1000, fsc)
        args = self.interface.args
        if sub_method == "longer_trajectories":
            args.num_steps = args.num_steps * 4
        if not hasattr(self, "pre_trainer"):
            logger.info("Creating pretrainer")
            self.pre_trainer = Actor_Value_Pretrainer(self.interface.environment, self.interface.tf_environment,
                                                     args, self.agent.agent.collect_data_spec)
            actor = self.pre_trainer.actor_net
            critic = self.pre_trainer.critic_net
            agent_folder = f"./trained_agents/{args.agent_name}_{args.learning_method}_{args.encoding_method}"
            self.agent = Recurrent_PPO_agent(self.interface.environment, self.interface.tf_environment, 
                                             args, args.load_agent, agent_folder, actor_net=actor, critic_net=critic)
        self.agent.evaluate_agent(vectorized=args.vectorized_envs_flag)
        if sub_method == "continuous_training":
            for _ in range(8):
                self.pre_trainer.train_both_networks(201, fsc=fsc, use_best_traj_only=False)
                self.agent.train_agent(500, vectorized=args.vectorized_envs_flag, replay_buffer_option=args.replay_buffer_option)
        else:
            self.pre_trainer.train_both_networks(nr_of_iterations // 4, fsc=fsc, use_best_traj_only=False)
            self.agent.train_agent(nr_of_iterations, vectorized=args.vectorized_envs_flag, replay_buffer_option=args.replay_buffer_option)

        # TODO: Other methods of training with FSC or SAYNT controller.

    def train_agent_with_jumpstarts(self, fsc, iterations):
        self.agent.train_agent(iterations, 
                               vectorized=self.interface.args.vectorized_envs_flag, 
                               replay_buffer_option=self.interface.args.replay_buffer_option, 
                               fsc=fsc,
                               jumpstart_fsc=True)
        
    def train_agent_with_shaping(self, fsc : FSC, iterations : int = 4000):
        self.agent.train_agent(iterations, 
                               vectorized=self.interface.args.vectorized_envs_flag, 
                               replay_buffer_option=self.interface.args.replay_buffer_option, 
                               fsc=fsc,
                               jumpstart_fsc=False,
                               shaping=True)
