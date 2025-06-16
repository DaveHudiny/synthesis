from agents.alternative_training.active_pretraining import EntropyRewardGenerator
from agents.recurrent_ppo_agent import Recurrent_PPO_agent
from tests.general_test_tools import init_args, init_environment

if __name__ == "__main__":
    # Example usage
    prism_model = "models/network-5-10-8/sketch.templ"
    prism_properties = "models/network-5-10-8/sketch.props"

    args = init_args(prism_model, prism_properties)
    env, tf_env = init_environment(args)
    agent = Recurrent_PPO_agent(env, tf_env, args)

    entropy_reward_generator = EntropyRewardGenerator(binary_flag=True, full_observability_flag=True, max_reward=1.0, decreaser='halve')

    agent.init_pretraining_driver(entropy_reward_generator)

    agent.reward_driver.run()
    # print(agent.replay_buffer.gather_all())
