from tools.args_emulator import ReplayBufferOptions
from environment.tf_py_environment import TFPyEnvironment
from tools.args_emulator import ArgsEmulator
from tools.evaluators import compute_average_return
from agents.father_agent import FatherAgent
import logging

logger = logging.getLogger(__name__)


class WeightInitializationMethods:

    class EvaluationSubResult:
        def __init__(self):
            self.cumulative_return = 0
            self.average_last_step_reward = 0

        def update(self, avg_return, avg_episode_return, reach_prob, episode_variance, num_episodes, trap_reach_prob, virtual_variance, combined_variance):
            self.cumulative_return = avg_return
            self.average_last_step_reward = avg_episode_return

    @staticmethod
    def _vectorized_evaluate_agent(agent: FatherAgent):
        """Evaluate the agent. Returns tuple of cumulative return and average last episode return.

        Args:
            agent (FatherAgent): The agent to evaluate.

        Returns:
            tuple: Tuple of cumulative return and average last episode return.
        """

        # Set number of evaluation environments to batch size, if the replay buffer is off-policy original and works only with a single environment for training.
        if agent.args.replay_buffer_option == ReplayBufferOptions.ORIGINAL_OFF_POLICY:
            agent.environment.set_num_envs(
                agent.args.batch_size)

        agent.tf_environment.reset()
        agent.vec_driver.run()

        if agent.args.replay_buffer_option == ReplayBufferOptions.ORIGINAL_OFF_POLICY:
            agent.environment.set_num_envs(1)
            agent.tf_environment.reset()

        subresult = WeightInitializationMethods.EvaluationSubResult()
        agent.trajectory_buffer.final_update_of_results(subresult.update)
        agent.trajectory_buffer.clear()
        cumulative_return = subresult.cumulative_return
        average_last_episode_return = subresult.average_last_step_reward
        return cumulative_return, average_last_episode_return

    @staticmethod
    def _evaluate_agent(agent: FatherAgent, vectorized: bool = False):
        """Evaluate the agent. Returns tuple of cumulative return and average last episode return.

        Args:
            agent (FatherAgent): The agent to evaluate.

        Returns:
            tuple: Tuple of cumulative return and average last episode return.
        """
        if not vectorized:
            cumulative_return, average_last_episode_return, _ = compute_average_return(
                agent.get_evaluation_policy(), agent.tf_environment, agent.args.evaluation_episodes)
        else:
            cumulative_return, average_last_episode_return = WeightInitializationMethods._vectorized_evaluate_agent(
                agent)

        return cumulative_return, average_last_episode_return

    @staticmethod
    def _check_saving_condition(cumulative_return, average_last_episode_return,
                                best_cumulative_return, best_average_last_episode_return):
        """Check if the agent should be saved as the best agent.

        Args:
            agent (FatherAgent): The agent to check.
            cumulative_return (float): The cumulative return of the agent.
            average_last_episode_return (float): The average last episode return of the agent.
            best_cumulative_return (float): The best cumulative return.
            best_average_last_episode_return (float): The best average last episode return.

        Returns:
            bool: True if the agent should be saved as the best agent, False otherwise.
        """
        if average_last_episode_return > best_average_last_episode_return:
            return True
        elif average_last_episode_return == best_average_last_episode_return:
            if cumulative_return > best_cumulative_return:
                return True
        return False

    @staticmethod
    def select_best_starting_weights(agent: FatherAgent, args: ArgsEmulator):
        logger.info("Selecting best starting weights")
        best_cumulative_return, best_average_last_episode_return = WeightInitializationMethods._evaluate_agent(
            agent, vectorized=args.vectorized_envs_flag)

        agent.save_agent()
        for i in range(args.restart_weights):
            logger.info(f"Restarting weights {i + 1}")
            agent.reset_weights()
            cumulative_return, average_last_episode_return = WeightInitializationMethods._evaluate_agent(
                agent, vectorized=args.vectorized_envs_flag)
            if WeightInitializationMethods._check_saving_condition(cumulative_return, average_last_episode_return,
                                                                   best_cumulative_return, best_average_last_episode_return):
                best_cumulative_return = cumulative_return
                best_average_last_episode_return = average_last_episode_return
                agent.save_agent()
        logger.info(f"Best cumulative return: {best_cumulative_return}")
        logger.info(
            f"Best average last episode return: {best_average_last_episode_return}")
        logger.info("Agent with best ")
        agent.load_agent()
        return agent
