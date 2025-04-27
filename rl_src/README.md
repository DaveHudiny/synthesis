# Implementation of RL
 Implementation of reinforcement learning for PAYNT oracle. Author of this implementation is David Hudák (xhudak03).
 
## Installation
 Install paynt.
 Then use following commands:
 ```shell
   $ source ../prerequisites/venv/bin/activate
   $ pip install libclang
   $ pip install tensorflow==2.15
   $ pip install tf_agents
   $ pip install tqdm dill matplotlib pandas seaborn
 ```

 This implementation was experimented within Ubuntu 22.04 and Debian 12.5. Other Linux distributions may miss some libraries etc. and you should install them on your own, or contact the authors (DaveHudiny at GitHub, or at my e-mail skolahudak@gmail.com, or one of the authors of PAYNT iandri@vutbr.cz).

## Used Framework and Sources
 The implementation is primarily based on TensorFlow Agents framework, which implements many important blocks of this project as reinforcement learning algorithms, TF environment interface, policy drivers etc. We also took some inspiration and in case of ./environment/pomdp_builder, we took the code from repository: [Shielding](https://github.com/stevencarrau/safe_RL_POMDPs).

## Examples of Run
 To run some selected model (intercept):
 ```shell
   $ source ../prerequisites/venv/bin/activate
   $ python3 experiment_runner.py --model-condition intercept
 ```

 To run experiments for multiple models from some_directory:
 ```shell
   $ source ../prerequisites/venv/bin/activate
   $ python3 experiment_runner.py --path-to-models some_directory
 ```
 
 If you want to design your own experiments, you should start with:
 ```shell
   $ source ../prerequisites/venv/bin/activate
   $ python3 experiment_runner.py --help
 ```
 
 The results of experiments are usually contained in folder ./experiments, where you can find various stuff like the results of training or pickle dictionaries for PAYNT (you can use PAYNT to use them as oracle).

 If you want to reload your old trained agents, they are by default located in trained\_agents. However, you should change the name in rl\_main.py, where to store agents, because all models are stored within the same folder by default (agent\_{algorithm\_name}), as we do not want to store multiple agents due to storage demands.
 
 There are also two files for plotting nice figures -- encoding\_plots.py for convergence curves of multiple experiments and plots.py for general learning curves. If you place them to folder with your results, for example from experiment_runner.py, then you can plot nice figures as in created diploma thesis.

## Known issues
 When compiling functions before evaluations, there occurs warning in form of:
 ```shell
  "/tmp/__autograph_generated_filew4d1jh49.py:14: SyntaxWarning: "is not" with a literal. Did you mean "!="?
   retval_ = ag\_\_.and\_(lambda : ag\_\_.ld(state) is not None, lambda : ag\_\_.and\_(lambda : ag\_\_.ld(state) is not (), lambda : ag\_\_.ld(state) is not []))"
   ```
 
 It is caused by compatibility issues of frameworks TF Agents and main TensorFlow, where we use TensorFlow graph compilation to speed up evaluation, as the TF Agents framework allows only graph optimization (utils.common.function) for train function and not for action and distribution functions. Instead, we optimize it with tf.function from TensorFlow, which throws this warning. It still provides significant speed-up of evaluation, but in future versions of TF Agents (if there exists), we recommend to use solely their function optimizations. The code is located in file agents/evaluators.py at lines 27-29.
