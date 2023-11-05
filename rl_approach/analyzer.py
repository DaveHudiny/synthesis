import rl_interface
import joblib

rl_interface.add_to_sys_argv_obstacle()
agent, storm_model, tf_env = rl_interface.shield_v2.improved_main()

with open("storm_model.pkl", "wb") as file:
    joblib.dump(storm_model, file)
