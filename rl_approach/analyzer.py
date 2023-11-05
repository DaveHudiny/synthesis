import rl_interface

rl_interface.add_to_sys_argv_obstacle()
agent, storm_model, tf_env = rl_interface.shield_v2.improved_main()

agent.save_model(path_to_model="./models_rl/obstacle_model_ppo_500runs.pkl")

tf_env.reset()
print(tf_env.current_time_step())