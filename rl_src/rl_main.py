# File: rl_main.py
# Description: Main for Reinforcement Learning Approach. If you want to train agents,
#              you can do it here, or you can use --reinforcement-learning option within PAYNT
# Author: David Hudak
# Login: xhudak03

import pickle
from agents.father_agent import FatherAgent
from agents.random_agent import RandomTFPAgent
from agents.policies.fsc_policy import FSC_Policy, FSC
from interpreters.tracing_interpret import TracingInterpret
from interpreters.model_free_interpret import ModelFreeInterpret, ModelInfo

from agents.recurrent_ppo_agent import Recurrent_PPO_agent
from agents.recurrent_ddqn_agent import Recurrent_DDQN_agent
from agents.recurrent_dqn_agent import Recurrent_DQN_agent
from agents.ppo_with_qvalues_fsc import PPO_with_QValues_FSC
from agents.periodic_fsc_neural_ppo import Periodic_FSC_Neural_PPO

import paynt.parser.sketch
import paynt.synthesizer.synthesizer_pomdp

from tf_agents.environments import parallel_py_environment
from tf_agents.environments import tf_py_environment
from tools.evaluators import *
from environment.environment_wrapper import *
from environment.pomdp_builder import *
from tools.args_emulator import ArgsEmulator
from tools.weight_initialization import WeightInitializationMethods 

import tensorflow as tf
import sys
import os
import rl_parser

import logging
import copy

logger = logging.getLogger(__name__)

logging.getLogger('tensorflow').setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

sys.path.append("../")


tf.autograph.set_verbosity(0)


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class PAYNT_Playground:
    @staticmethod
    def fill_nones_in_qvalues(qvalues):
        for state in range(len(qvalues)):
            for memory in range(len(qvalues[state])):
                if qvalues[state][memory] is None:
                    qvalues[state][memory] = np.mean([qvalues[state][i] for i in range(
                        len(qvalues[state])) if qvalues[state][i] is not None])
        return qvalues

    @classmethod # Not a good implementation, if we work in a loop with multiple different models.
    def singleton_init_models(cls, sketch_path, properties_path):
        if not os.path.exists(sketch_path):
            raise ValueError(f"Sketch file {sketch_path} does not exist.")
        if not hasattr(cls, "quotient") and not hasattr(cls, "synthesizer"):
            cls.quotient = paynt.parser.sketch.Sketch.load_sketch(
                sketch_path, properties_path)
            cls.k = 3  # May be unknown?
            cls.quotient.set_imperfect_memory_size(cls.k)
            cls.synthesizer = paynt.synthesizer.synthesizer_pomdp.SynthesizerPomdp(
                cls.quotient, method="ar", storm_control=None)

    @classmethod
    def compute_qvalues_function(cls):
        assignment = cls.synthesizer.synthesize()
        # before the quotient is modified we can use this assignment to compute Q-values
        assert assignment is not None, "Provided assignment cannot be None."
        qvalues = cls.quotient.compute_qvalues(assignment)
        # note Q(s,n) may be None if (s,n) exists in the unfolded POMDP but is not reachable in the induced DTMC
        memory_size = len(qvalues[0])
        assert cls.k == memory_size
        qvalues = PAYNT_Playground.fill_nones_in_qvalues(qvalues)
        return qvalues

    @classmethod
    def get_fsc_critic_components(cls, sketch_path, properties_path):
        cls.singleton_init_models(
            sketch_path=sketch_path, properties_path=properties_path)
        qvalues = cls.compute_qvalues_function()
        action_labels_at_observation = cls.quotient.action_labels_at_observation
        return qvalues, action_labels_at_observation


def save_dictionaries(name_of_experiment, model, learning_method, refusing_typ, obs_action_dict, memory_dict, labels):
    """ Save dictionaries for Paynt oracle.
    Args:
        name_of_experiment (str): Name of the experiment.
        model (str): The name of the model.
        learning_method (str): The learning method.
        refusing_typ (str): Whether to use refusing when interpreting.
        obs_action_dict (dict): The observation-action dictionary.
        memory_dict (dict): The memory dictionary.
        labels (dict): The labels dictionary.
    """
    if not os.path.exists(f"{name_of_experiment}/{model}_{learning_method}/{refusing_typ}"):
        os.makedirs(
            f"{name_of_experiment}/{model}_{learning_method}/{refusing_typ}")
    with open(f"{name_of_experiment}/{model}_{learning_method}/{refusing_typ}/obs_action_dict.pickle", "wb") as f:
        pickle.dump(obs_action_dict, f)
    with open(f"{name_of_experiment}/{model}_{learning_method}/{refusing_typ}/memory_dict.pickle", "wb") as f:
        pickle.dump(memory_dict, f)
    with open(f"{name_of_experiment}/{model}_{learning_method}/{refusing_typ}/labels.pickle", "wb") as f:
        pickle.dump(labels, f)


def save_statistics_to_new_json(name_of_experiment, model, learning_method, evaluation_result: EvaluationResults, args: dict = None):
    """ Save statistics to a new JSON file.
    Args:
        name_of_experiment (str): Name of the experiment.
        model (str): The name of the model.
        learning_method (str): The learning method.
        evaluation_result (EvaluationResults): The evaluation results.
        args (dict, optional): The arguments. Defaults to None.
    """
    if args is None:
        max_steps = 300
    else:
        max_steps = args.max_steps

    evaluation_result.set_experiment_settings(
        learning_algorithm=learning_method, max_steps=max_steps)
    if not os.path.exists(f"{name_of_experiment}"):
        os.mkdir(f"{name_of_experiment}")
    if os.path.exists(f"{name_of_experiment}/{model}_{learning_method}_training.json"):
        i = 1
        while os.path.exists(f"{name_of_experiment}/{model}_{learning_method}_training_{i}.json"):
            i += 1
        evaluation_result.save_to_json(
            f"{name_of_experiment}/{model}_{learning_method}_training_{i}.json")
    else:
        evaluation_result.save_to_json(
            f"{name_of_experiment}/{model}_{learning_method}_training.json")


class Initializer:
    def __init__(self, args: ArgsEmulator = None, pomdp_model=None, agent=None):
        if args is None:
            self.parser = rl_parser.Parser()
            self.args = self.parser.args
        else:
            self.args = args
        self.pomdp_model = pomdp_model
        self.agent = agent

    def asserts(self):
        if self.args.prism_model and not self.args.prism_properties:
            raise ValueError("Prism model is set but Prism properties are not")
        if self.args.paynt_fsc_imitation and not self.args.paynt_fsc_json:
            raise ValueError(
                "Paynt imitation is set but there is not selected any JSON FSC file.")

    def parse_properties(self):
        with open(self.args.prism_properties, "r") as f:
            lines = f.readlines()
        properties = []
        for line in lines:
            if line.startswith("//"):
                continue
            properties.append(line.strip())
        return properties

    def initialize_prism_model(self):
        properties = self.parse_properties()
        pomdp_args = POMDP_arguments(
            self.args.prism_model, properties, self.args.constants)
        return POMDP_builder.build_model(pomdp_args)

    def run_agent(self):
        num_steps = 10
        for _ in range(num_steps):
            time_step = self.tf_environment._reset()
            is_last = time_step.is_last()
            while not is_last:
                action_step = self.agent.policy(time_step)
                next_time_step = self.tf_environment.step(action_step.action)
                time_step = next_time_step
                is_last = time_step.is_last()

    def initialize_environment(self, parallelized: bool = False, num_parallel_environments: int = 4, random_init_simulator : bool = False):
        self.pomdp_model = self.initialize_prism_model()
        logger.info("Model initialized")
        if random_init_simulator:
            sketch_path = self.args.prism_model
            props_path = self.args.prism_properties
            # qvalues_table = PAYNT_Playground.compute_qvalues_function(sketch_path, props_path)
            # qvalues_table, action_labels_at_observation = PAYNT_Playground.get_fsc_critic_components(
            #     sketch_path, props_path)
            qvalues_table = None
            self.second_pomdp_model = self.initialize_prism_model() # Second instance of StormPy model
            self.args.random_start_simulator = False
            rand_args = copy.deepcopy(self.args)
            rand_args.random_start_simulator = True
            self.environment = {"eval_model": Environment_Wrapper(self.pomdp_model, self.args),
                                "train_model": Environment_Wrapper(self.second_pomdp_model, rand_args, qvalues_table)}
        else:
            self.environment = Environment_Wrapper(self.pomdp_model, self.args)
        if parallelized:
            tf_environment = parallel_py_environment.ParallelPyEnvironment(
                [self.environment.create_new_environment] * num_parallel_environments)
        else:
            if random_init_simulator:
                tf_environment = {"eval_sim": tf_py_environment.TFPyEnvironment(self.environment["eval_model"]),
                                  "train_sim": tf_py_environment.TFPyEnvironment(self.environment["train_model"])}
            else:
                tf_environment = tf_py_environment.TFPyEnvironment(
                        self.environment)
        logger.info("Environment initialized")
        return tf_environment

    

    def select_agent_type(self, learning_method=None, qvalues_table=None, action_labels_at_observation=None,
                          pre_training_dqn : bool = False) -> FatherAgent:
        """Selects the agent type based on the learning method and encoding method in self.args. The agent is saved to the self.agent variable.

        Args:
            learning_method (str, optional): The learning method. If set, the learning method is used instead of the one from the args object. Defaults to None.
            qvalues_table (dict, optional): The Q-values table created by the product of POMDPxFSC. Defaults to None.
        Raises:
            ValueError: If the learning method is not recognized or implemented yet."""
        if learning_method is None:
            learning_method = self.args.learning_method
        agent_folder = f"./trained_agents/{self.args.agent_name}_{self.args.learning_method}_{self.args.encoding_method}"
        if learning_method == "DQN":
            agent = Recurrent_DQN_agent(
                self.environment, self.tf_environment, self.args, load=self.args.load_agent, agent_folder=agent_folder,
                single_value_qnet=pre_training_dqn)
        elif learning_method == "DDQN":
            agent = Recurrent_DDQN_agent(
                self.environment, self.tf_environment, self.args, load=self.args.load_agent, agent_folder=agent_folder)
        elif learning_method == "PPO":
            agent = Recurrent_PPO_agent(
                self.environment, self.tf_environment, self.args, load=self.args.load_agent, agent_folder=agent_folder)
        elif learning_method == "Stochastic_PPO":
            self.args.prefer_stochastic = True
            agent = Recurrent_PPO_agent(
                self.environment, self.tf_environment, self.args, load=self.args.load_agent, agent_folder=agent_folder)
        elif learning_method == "PPO_FSC_Critic":
            if qvalues_table is None:  # If Q-values table is not provided, compute it from the sketch and properties
                sketch_path = self.args.prism_model
                props_path = self.args.prism_properties
                # qvalues_table = PAYNT_Playground.compute_qvalues_function(sketch_path, props_path)
                qvalues_table, action_labels_at_observation = PAYNT_Playground.get_fsc_critic_components(
                    sketch_path, props_path)
            assert action_labels_at_observation is not None  # Action labels must be provided
            agent = PPO_with_QValues_FSC(
                self.environment, self.tf_environment, self.args, load=self.args.load_agent, agent_folder=agent_folder,
                qvalues_table=qvalues_table, action_labels_at_observation=action_labels_at_observation)
        elif learning_method == "Periodic_FSC_Neural_PPO":
            if qvalues_table is None:  # If Q-values table is not provided, compute it from the sketch and properties
                sketch_path = self.args.prism_model
                props_path = self.args.prism_properties
                qvalues_table, action_labels_at_observation = PAYNT_Playground.get_fsc_critic_components(
                    sketch_path, props_path)
                assert action_labels_at_observation is not None  # Action labels must be provided
            agent = Periodic_FSC_Neural_PPO(
                self.environment, self.tf_environment, self.args, load=self.args.load_agent, agent_folder=agent_folder,
                qvalues_table=qvalues_table, action_labels_at_observation=action_labels_at_observation)
        else:
            raise ValueError(
                "Learning method not recognized or implemented yet.")
        return agent
    
    def get_tf_environment_eval(self):
        if isinstance(self.tf_environment, dict):
            return self.tf_environment["eval_sim"]
        else:
            return self.tf_environment
        
    def get_tf_environment_train(self):
        if isinstance(self.tf_environment, dict):
            return self.tf_environment["train_sim"]
        else:
            return self.tf_environment


    def initialize_agent(self, qvalues_table=None, action_labels_at_observation=None, learning_method = None,
                         pre_training_dqn : bool = False) -> FatherAgent:
        """Initializes the agent. The agent is initialized based on the learning method and encoding method. The agent is saved to the self.agent variable.
        It is important to have previously initialized self.environment, self.tf_environment and self.args.

        returns:
            FatherAgent: The initialized agent.
        """
        agent = self.select_agent_type(
            qvalues_table=qvalues_table, action_labels_at_observation=action_labels_at_observation,
            learning_method=learning_method, pre_training_dqn = pre_training_dqn)
        if self.args.restart_weights > 0:
            tf_environment = self.get_tf_environment_eval()
            agent = WeightInitializationMethods.select_best_starting_weights(agent, tf_environment, self.args)
        return agent

    def initialize_fsc_agent(self):
        with open("FSC_experimental.json", "r") as f:
            fsc_json = json.load(f)
        fsc = FSC.from_json(fsc_json)
        action_keywords = self.environment.action_keywords
        policy = FSC_Policy(self.tf_environment, fsc,
                            tf_action_keywords=action_keywords)
        return policy

    def evaluate_random_policy(self):
        agent = RandomTFPAgent(
            self.environment, self.tf_environment, self.args, load=False)
        interpret = TracingInterpret(self.environment, self.tf_environment,
                                     self.args.encoding_method, self.environment._possible_observations)
        results = {}
        for refusing in [True, False]:
            result = interpret.get_dictionary(agent, refusing)
            if refusing:
                results["best_with_refusing"] = result
                results["last_with_refusing"] = result
            else:
                results["best_without_refusing"] = result
                results["last_without_refusing"] = result
        return results

    def tracing_interpretation(self, with_refusing=None):
        interpret = TracingInterpret(self.environment, self.tf_environment,
                                     self.args.encoding_method)
        for quality in ["last", "best"]:
            logger.info(f"Interpreting agent with {quality} quality")
            if with_refusing == None:
                result = {}
                self.agent.load_agent(quality == "best")
                if self.args.learning_method == "PPO" and self.args.prefer_stochastic:
                    self.agent.set_agent_stochastic()
                elif self.args.learning_method == "PPO":
                    self.agent.set_agent_greedy()
                result[f"{quality}_with_refusing"] = interpret.get_dictionary(
                    self.agent, with_refusing=True)
                result[f"{quality}_without_refusing"] = interpret.get_dictionary(
                    self.agent, with_refusing=False)
            else:
                result = interpret.get_dictionary(self.agent, with_refusing)
        return result

    def main(self, with_refusing=False):
        try:
            self.asserts()
        except ValueError as e:

            logger.error(e)
            return
        self.tf_environment = self.initialize_environment(random_init_simulator=self.args.random_start_simulator)
        if self.args.evaluate_random_policy:  # Evaluate random policy
            return self.evaluate_random_policy()

        self.agent = self.initialize_agent()
        if self.args.learning_method == "PPO" and self.args.set_ppo_on_policy:
            self.agent.train_agent_on_policy(self.args.nr_runs)
        else:
            self.agent.train_agent_off_policy(self.args.nr_runs)
        self.agent.save_agent()
        result = {}
        logger.info("Training finished")
        if self.args.interpretation_method == "Tracing":
            result = self.tracing_interpretation(with_refusing)
        else:
            raise ValueError(
                "Interpretation method not recognized or implemented yet.")
        return result

    def __del__(self):
        if hasattr(self, "tf_environment") and self.tf_environment is not None:
            try:
                self.tf_environment.close()
            except Exception as e:
                pass
        if hasattr(self, "agent") and self.agent is not None:
            self.agent.save_agent()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initializer = Initializer()
    args = initializer.parser.args
    result = initializer.main(args.with_refusing)
    if args.with_refusing is None:
        save_dictionaries(args.experiment_directory, args.agent_name,
                          args.learning_method, "best_with_refusing", result["best_with_refusing"][0], result["best_with_refusing"][1], result["best_with_refusing"][2])

        save_dictionaries(args.experiment_directory, args.agent_name,
                          args.learning_method, "last_with_refusing",
                          result["last_with_refusing"][0], result["last_with_refusing"][1],
                          result["last_with_refusing"][2])
        save_statistics_to_new_json(args.experiment_directory, args.agent_name, args.learning_method,
                                    initializer.agent.evaluation_result, args)
        save_dictionaries(args.experiment_directory, args.agent_name,
                          args.learning_method, "best_without_refusing", result[
                              "best_without_refusing"][0],
                          result["best_without_refusing"][1], result["best_without_refusing"][2])
        save_dictionaries(args.experiment_directory, args.agent_name,
                          args.learning_method, "last_without_refusing", result[
                              "last_without_refusing"][0],
                          result["last_without_refusing"][1], result["last_without_refusing"][2])
    else:
        save_dictionaries(args.experiment_directory, args.agent_name,
                          args.learning_method, args.with_refusing, result[0], result[1], result[2])
        save_statistics_to_new_json(args.experiment_directory, args.agent_name, args.learning_method,
                                    initializer.agent.evaluation_result, args)
