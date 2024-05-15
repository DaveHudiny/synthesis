# Implementation of RL
 Implementation of reinforcement learning for PAYNT oracle. 
 
## Installation
 Install paynt.
 Then use following commands:
   $ source ../prerequisites/venv/bin/activate
   $ pip install tensorflow==2.15
   $ pip install tf_agents
   $ pip install tqdm dill matplotlib pandas seaborn

 This implementation was experimented within Ubuntu 22.04 and Debian 12.5. Other Linux distributions may miss some libraries etc. and you should install them on your own, or contact the authors (DaveHudiny at GitHub, or at my e-mail skolahudak@gmail.com, or one of the authors of PAYNT iandri@vutbr.cz).

## Used Framework and Sources
 The implementation is primarily based on TensorFlow Agents framework, which implements many important blocks of this project as reinforcement learning algorithms, TF environment interface, policy drivers etc. We also took some inspiration and in case of ./environment/pomdp_builder, we took the code from repository: [Shielding](https://github.com/stevencarrau/safe_RL_POMDPs).

## Examples of Run
 To run some selected model (intercept):
   $ source ../prerequisites/venv/bin/activate
   $ python3 rl_main.py --prism-model ./models/intercept/sketch.templ --prism-properties ./models/intercept/sketch.props
 
 To run experiments for multiple models:
   $ source ../prerequisites/venv/bin/activate
   $ python3 experiment_runner.py
 
 If you want to design your own experiments, you should start with:
   $ source ../prerequisites/venv/bin/activate
   $ python3 rl_main.py --help
 
 The results of experiments are usually contained in folder ./experiments, where you can find various stuff like the results of training or pickle dictionaries for PAYNT (you can use PAYNT to use them as oracle).

 If you want to reload your old trained agents, they are by default located in trained_agents. However, you should change the name in rl_main.py, where to store agents, because all models are stored within the same folder by default (agent_{algorithm_name}), as we do not want to store multiple agents due to storage demands.