import logging

logger = logging.getLogger(__name__)

from agents.father_agent import FatherAgent
from tools.evaluators import compute_average_return
from tools.args_emulator import ArgsEmulator

from tf_agents.environments.tf_py_environment import TFPyEnvironment

class WeightInitializationMethods:
    
    @staticmethod
    def select_best_starting_weights(agent: FatherAgent, tf_environment : TFPyEnvironment, args : ArgsEmulator):
        logger.info("Selecting best starting weights")
        best_cumulative_return, best_average_last_episode_return, _ = compute_average_return(
            agent.get_evaluation_policy(), tf_environment, args.evaluation_episodes)
        agent.save_agent()
        for i in range(args.restart_weights):
            logger.info(f"Restarting weights {i + 1}")
            agent.reset_weights()
            cumulative_return, average_last_episode_return, _ = compute_average_return(
                agent.get_evaluation_policy(), tf_environment, args.evaluation_episodes)
            if average_last_episode_return > best_average_last_episode_return:
                best_cumulative_return = cumulative_return
                best_average_last_episode_return = average_last_episode_return
                agent.save_agent()
            elif average_last_episode_return == best_average_last_episode_return:
                if cumulative_return > best_cumulative_return:
                    best_cumulative_return = cumulative_return
                    agent.save_agent()
        logger.info(f"Best cumulative return: {best_cumulative_return}")
        logger.info(
            f"Best average last episode return: {best_average_last_episode_return}")
        logger.info("Agent with best ")
        agent.load_agent()
        return agent
