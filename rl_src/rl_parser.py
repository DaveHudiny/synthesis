# File: rl_parser.py
# Description: Parser for Reinforcement Learning Approach
# Author: David Hudák

import argparse

interpetation_methods = ["Model-Free", "Tracing", "Both"]
encoding_observation_methods = ["One-Hot", "Integer", "Valuations"]


class Parser:
    """Parser for Reinforcement Learning Approach. It uses argparse library to parse the arguments. 
    The arguments are divided into several groups. The parser is used in rl_initializer.py.
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Reinforcement Learning Approach Parser")
        self.environment_group = self.parser.add_argument_group(
            'Environment Parameters')
        self.agent_group = self.parser.add_argument_group('Agent Parameters')
        self.learning_group = self.parser.add_argument_group(
            'Learning Parameters')
        self.logger_group = self.parser.add_argument_group('Logger Parameters')
        self.evaluation_group = self.parser.add_argument_group(
            'Evaluation Parameters')
        self.interpretation_group = self.parser.add_argument_group(
            'Interpretation Parameters')

        self.environment_parser()
        self.agent_parameters_parser()
        self.learning_parameters_parser()
        self.logger_parser()
        self.evaluation_parser()
        self.interpretation_method_parser()
        self.args = self.parser.parse_args()

    def environment_parser(self):
        """Parses the environment parameters. The parameters are added to the environment group, which describes information for environment (model) generation."""
        self.environment_group.add_argument("--grid-model", type=str,
                                            help="Grid model to be used", choices=["avoid", "evade", "intercept", "obstacle", "refuel", "rocks"])
        self.environment_group.add_argument("--prism-model", type=str, default="./prism_models/obstacle/sketch.templ",
                                            help="Select file with prism model")
        self.environment_group.add_argument(
            "--prism-properties", type=str, default="./prism_models/obstacle/sketch.props", help="Select file with prism properties")
        self.environment_group.add_argument(
            "--constants", type=str, default="", help="Constants for the model. Syntax looks like: \"C1=10,C2=60\". See Prism template for definable constants")
        self.environment_group.add_argument(
            "--discount-factor", type=float, default=0.9, help="Discount factor for the environment")
        self.environment_group.add_argument(
            "--encoding-method", type=str, default="One-Hot", help="Encoding method for the observations", choices=encoding_observation_methods)
        self.environment_group.add_argument(
            "--action-filtering", action="store_true",
            help="Filtering of actions performed by the environment. If set, the environment will filter the actions based on the current state and return negative reward.")
        self.environment_group.add_argument(
            "illegal-action-penalty", type=float, default=-1, help="Penalty for illegal actions")
        self.environment_group.add_argument(
            "--randomizing-illegal-actions", action="store_true", help="Randomize illegal actions")
        self.environment_group.add_argument(
            "--randomizing-penalty", type=float, default=-0.1, help="Penalty for randomizing illegal actions")
        self.environment_group.add_argument(
            "--reward-shaping", action="store_true", help="Reward shaping")
        self.environment_group.add_argument(
            "--reward-shaping-model", type=str, help="Reward shaping model", choices=["evade", "refuel"])

    def agent_parameters_parser(self):
        """Parses the agent parameters. The parameters are added to the agent group, which describes information for agent generation."""
        self.agent_group.add_argument("--learning-method", type=str,
                                      default="DQN", choices=["DQN"], help="Learning method")
        self.agent_group.add_argument(
            "--agent-folder", type=str, help="Agent folder")
        self.agent_group.add_argument(
            "--using_logits", action="store_true", help="Using logits, compatibile with PPO.")
        self.agent_group.add_argument(
            "--paynt-fsc-imitation", action="store_true", help="Use extracted FSC from Paynt for improving data collection and imitation learning")
        self.agent_group.add_argument(
            "--paynt-fsc-json", type=str, help="JSON file with extracted FSC from Paynt")

    def learning_parameters_parser(self):
        """Parses the learning parameters. The parameters are added to the learning group, which describes information for learning process."""

        self.learning_group.add_argument("--learning-rate", type=float, default=1e-7,
                                         help="Learning rate")
        self.learning_group.add_argument(
            "--max-steps", type=int, default=100, help="Maximum steps per episode")
        self.learning_group.add_argument(
            "--batch-size", type=int, default=64, help="Batch size")
        self.learning_group.add_argument(
            "--num-steps", type=int, default=25, help="Number of steps for each sample trajectory. Used for training the agent.")
        self.learning_group.add_argument(
            "--nr-runs", type=int, default=5000, help="Number of runs")
        self.learning_group.add_argument(
            "--seed", type=int, default=123456, help="Seed for reproducibility")
        self.learning_group.add_argument(
            "--buffer-size", type=int, default=1000, help="Buffer size for the replay buffer")
        self.learning_group.add_argument(
            "--restart-weights", type=int, default=0, help="Number of restarts of weights before starting learning")
        self.learning_group.add_argument(
            "--fsc-policy-max-iteration", type=int, default=100, help="If --paynt-fsc-imitation is selected, this parameter defines the maximum number of iterations for FSC policy training.")

    def logger_parser(self):
        """Parses the logger parameters. The parameters are added to the logger group, which describes information for logging."""
        self.logger_group.add_argument(
            "--log-dir", type=str, default="logs", help="Log directory")
        self.logger_group.add_argument(
            "--log-filename", type=str, default="logs.txt", help="Log filename")
        self.logger_group.add_argument(
            "--experiment-directory", type=str, default="experiments", help="Directory for files from experiments.")
        self.logger_group.add_argument(
            "--save-model-drn", type=str, help="Save environment model to drn file. If set, the file will be saved to the specified file.")
        self.logger_group.add_argument(
            "--save-agent", action="store_true", help="Save agent model during training")
        self.logger_group.add_argument(
            "--load-agent", action="store_true", help="Load agent model during training")

    def evaluation_parser(self):
        """Parses the evaluation parameters. The parameters are added to the evaluation group, which describes information for evaluation."""
        self.evaluation_group.add_argument(
            "--evaluation-episodes", type=int, default=10, help="Number of episodes for evaluation")
        self.evaluation_group.add_argument(
            "--evaluation-goal", type=float, default=10, help="Reward for reaching goal.")
        self.evaluation_group.add_argument(
            "--evaluation-antigoal", type=float, default=-10, help="Reward for not reaching goal.")

    def interpretation_method_parser(self):
        """Parses the interpretation parameters. The parameters are added to the interpretation group, which describes information for interpretation."""
        self.interpretation_group.add_argument(
            "--interpretation-method", type=str, default="Tracing", help="Interpretation method", choices=interpetation_methods)
        self.interpretation_group.add_argument("--interpretation-granularity", type=int, default=20,
                                               help="Defines the granularity of the interpretation method (e.g. number of evaluations for each state for Monte-Carlo method)")


if __name__ == "__main__":
    parser = Parser()
    print(parser.args)
