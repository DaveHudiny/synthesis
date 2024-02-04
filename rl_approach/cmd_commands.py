import sys

class CmdCommands:

    def add_to_sys_argv_obstacle():
            sys.argv.append("-m")
            sys.argv.append("obstacle")
            sys.argv.append("--constants")
            sys.argv.append("N=6")
        
    def add_to_sys_argv_refuel():
            sys.argv.append("-m")
            sys.argv.append("refuel")
            sys.argv.append("-c")
            sys.argv.append("N=6,ENERGY=10")
        
    def add_to_sys_argv_intercept():
            sys.argv.append("-m")
            sys.argv.append("intercept")
            sys.argv.append("-c")
            sys.argv.append("N=5,RADIUS=2")

    def add_to_sys_argv_evade():
            sys.argv.append("-m")
            sys.argv.append("evade")
            sys.argv.append("-c")
            sys.argv.append("N=6,RADIUS=2")
        
    def add_to_sys_argv_avoid():
            sys.argv.append("-m")
            sys.argv.append("avoid")
            sys.argv.append("-c")
            sys.argv.append("N=6")
        
    def add_to_sys_argv_rocks():
            sys.argv.append("-m")
            sys.argv.append("rocks")
            sys.argv.append("-c")
            sys.argv.append("N=6")

    def add_to_sys_argv_strategy(model):
            sys.argv.append("--learning_method")
            sys.argv.append(f"{model}")

    def add_to_sys_argv_num_episodes(num_episodes):
            sys.argv.append("--eval-episodes")
            sys.argv.append(str(num_episodes))
