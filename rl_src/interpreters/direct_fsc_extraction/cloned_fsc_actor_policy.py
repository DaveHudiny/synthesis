import tf_agents
import tensorflow as tf
import numpy as np
import keras
from keras import backend as K

from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.environments.tf_py_environment import TFPyEnvironment

from keras import optimizers

from interpreters.direct_fsc_extraction.fsc_like_actor_network import FSCLikeActorNetwork
from tools.evaluation_results_class import EvaluationResults
from tools.specification_check import SpecificationChecker
from tools.args_emulator import ArgsEmulator
from environment.environment_wrapper_vec import EnvironmentWrapperVec
from tools.evaluators import evaluate_policy_in_model
from interpreters.direct_fsc_extraction.extraction_stats import ExtractionStats



import logging

import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClonedFSCActorPolicy(TFPolicy):
    def __init__(self, original_policy: TFPolicy, memory_size: int,
                 observation_and_action_constrint_splitter=None,
                 use_one_hot=True,
                 use_residual_connection=True,
                 optimization_specification: SpecificationChecker.Constants = SpecificationChecker.Constants.REACHABILITY,
                 model_name: str = "generic_model",
                 find_best_policy: bool = False,
                 max_episode_length: int = 1000):
        self.original_policy = original_policy
        self.use_one_hot = use_one_hot
        policy_state_spec = BoundedArraySpec(
            (memory_size,), np.float32, minimum=-1.5, maximum=1.5)
        super(ClonedFSCActorPolicy, self).__init__(time_step_spec=original_policy.time_step_spec,
                                                   action_spec=original_policy.action_spec,
                                                   policy_state_spec=policy_state_spec,
                                                   observation_and_action_constraint_splitter=observation_and_action_constrint_splitter)
        self.memory_size = memory_size
        self.fsc_actor = FSCLikeActorNetwork(
            original_policy.time_step_spec.observation["observation"].shape,
            original_policy.action_spec.maximum + 1,
            memory_size,
            use_one_hot=use_one_hot,
            use_residual_connection=use_residual_connection)
        self.model_name = model_name
        self.optimization_specification = optimization_specification
        self.find_best_policy = find_best_policy
        self.max_episode_length = max_episode_length

    def load_best_policy(self):
        try:
            self.fsc_actor.load_weights(
                f"experiments_models/{self.model_name}/best_policy.h5")
        except Exception as e:
            logger.error(f"Could not load best policy: {e}")

    def save_best_policy(self):
        os.makedirs(os.path.dirname(f"experiments_models/{self.model_name}/best_policy.h5"), exist_ok=True)
        self.fsc_actor.save_weights(
            f"experiments_models/{self.model_name}/best_policy.h5")

    def _variables(self):
        return self.fsc_actor.variables

    def distro(self, time_step, policy_state, seed):
        observation, mask = self.observation_and_action_constraint_splitter(
            time_step.observation)
        observation = tf.reshape(observation, (observation.shape[0], 1, -1))
        policy_state = tf.reshape(policy_state, (policy_state.shape[0], -1))
        step_type = tf.reshape(time_step.step_type,
                               (time_step.step_type.shape[0], 1, -1))
        step_type = tf.cast(step_type, tf.float32)
        action, memory = self.fsc_actor(
            observation, step_type, policy_state)
        action = tf.reshape(action, (action.shape[0], -1))
        # Change logits of illegal actions to -inf
        action = tf.where(mask, action, -1e20)
        # memory = memory[:, -1, :]
        return action, memory

    def _action(self, time_step, policy_state, seed):
        action_probs, memory = self.distro(
            time_step, policy_state, seed)
        # action = tf.random.categorical(action_probs, 1, dtype=tf.int32)
        # sample the most probable action
        action = tf.argmax(action_probs, axis=-1, output_type=tf.int32)
        action = tf.reshape(action, (action.shape[0],))
        policy_step = PolicyStep(action=action, state=memory)
        # print(policy_step)
        return policy_step

    def _get_initial_state(self, batch_size: int):
        init_state = self.fsc_actor.get_initial_state(batch_size)
        return tf.reshape(init_state, (batch_size, -1))

    def behavioral_clone_original_policy_to_fsc(self, buffer: TFUniformReplayBuffer, num_epochs: int,
                                                sample_len=32,
                                                specification_checker: SpecificationChecker = None,
                                                environment: EnvironmentWrapperVec = None,
                                                tf_environment: TFPyEnvironment = None,
                                                args: ArgsEmulator = None,
                                                extraction_stats: ExtractionStats = None) -> ExtractionStats:
        cloned_actor = self
        dataset_options = tf.data.Options()
        dataset_options.experimental_deterministic = False
        dataset = (
            buffer.as_dataset(sample_batch_size=64, num_steps=sample_len,
                              num_parallel_calls=tf.data.AUTOTUNE)
            .with_options(dataset_options)
            .prefetch(tf.data.AUTOTUNE)
        )

        if extraction_stats is None:
            extraction_stats = ExtractionStats(
                original_policy_reachability=0,
                original_policy_reward=0,
                use_one_hot=self.use_one_hot,
                number_of_samples=sample_len,
                memory_size=self.memory_size,
                residual_connection=self.fsc_actor.use_residual_connection
            )

        iterator = iter(dataset)
        neural_fsc = cloned_actor.fsc_actor
        optimizer = optimizers.Adam(learning_rate=1.6e-4, weight_decay=1e-3)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        loss_metric = keras.metrics.Mean(name="train_loss")
        accuracy_metric = keras.metrics.SparseCategoricalAccuracy(
            name="accuracy")

        self.evaluation_result = None

        @tf.function
        def train_step(experience):
            observations = experience.observation["observation"]
            gt_actions = experience.action
            step_types = tf.cast(experience.step_type, tf.float32)
            step_types = tf.reshape(step_types, (step_types.shape[0], -1, 1))
            with tf.GradientTape() as tape:
                played_action, mem = neural_fsc(
                    observations, step_type=step_types)
                loss = loss_fn(gt_actions, played_action)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, neural_fsc.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(
                zip(grads, neural_fsc.trainable_variables))
            loss_metric.update_state(loss)
            accuracy_metric.update_state(gt_actions, played_action)
            return loss
        
        @tf.function
        def train_step_by_step(experience):
            observations = experience.observation["observation"]
            gt_actions = experience.action
            step_types = tf.cast(experience.step_type, tf.float32)
            loss = []
            with tf.GradientTape() as tape:
                mem = neural_fsc.get_initial_state(
                    observations.shape[0])
                mem = tf.reshape(mem, (mem.shape[0], -1))
                for i in range(step_types.shape[1]):
                    observation = tf.reshape(observations[:, i], (observations.shape[0], 1, -1))
                    step_type = tf.reshape(step_types[:, i], (step_types.shape[0], 1, -1))
                    played_action, mem = neural_fsc(
                        observation, step_type=step_type, old_memory=mem)
                    played_action = tf.reshape(played_action, (played_action.shape[0], -1))
                    mem = tf.where(tf.equal(step_type, tf_agents.trajectories.time_step.StepType.LAST), 
                                   neural_fsc.get_initial_state(observations.shape[0]), mem)
                    loss.append(loss_fn(gt_actions[:, i], played_action))
                    accuracy_metric.update_state(gt_actions[:, i], played_action)
                loss = tf.reduce_mean(loss)
            # loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, neural_fsc.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(
                zip(grads, neural_fsc.trainable_variables))
            loss_metric.update_state(loss)
            
            return loss

        for i in range(num_epochs):
            try:
                experience, _ = next(iterator)
            except StopIteration:  # Reset iteratoru, pokud dojde dataset
                iterator = iter(dataset)
                experience, _ = next(iterator)

            loss = train_step(experience) # train_step(experience)
            self.periodical_evaluation(i, loss_metric, accuracy_metric, cloned_actor,
                                       environment, tf_environment, extraction_stats,
                                       self.evaluation_result, specification_checker)
            # if i > 10000 and accuracy_metric.result() > 0.98:
            #     break
            

        return extraction_stats

    def periodical_evaluation(self,
                              iteration_number: int,
                              loss_metric: keras.metrics.Mean,
                              accuracy_metric: keras.metrics.SparseCategoricalAccuracy,
                              cloned_actor: TFPolicy,
                              environment: EnvironmentWrapperVec,
                              tf_environment: TFPyEnvironment,
                              extraction_stats: ExtractionStats,
                              evaluation_result: EvaluationResults,
                              specification_checker: SpecificationChecker):
        if iteration_number % 1000 == 0:
            avg_loss = loss_metric.result()
            accuracy = accuracy_metric.result()
            logger.info(f"Epoch {iteration_number}, Loss: {avg_loss:.4f}")
            logger.info(f"Epoch {iteration_number}, Accuracy: {accuracy:.4f}")
            extraction_stats.add_evaluation_accuracy(accuracy)
            loss_metric.reset_states()
            accuracy_metric.reset_states()
            
        if iteration_number % 5000 == 0:
            self.evaluation_result = evaluate_policy_in_model(
                cloned_actor, None, environment, tf_environment, self.max_episode_length * 2, evaluation_result)
            extraction_stats.add_extraction_result(
                self.evaluation_result.reach_probs[-1], self.evaluation_result.returns[-1])
            if False and specification_checker is not None:
                if specification_checker.check_specification(evaluation_result.reach_probs[-1], evaluation_result.returns[-1]):
                    pass
            self.check_and_save(self.evaluation_result)

    def check_and_save(self, evaluation_result: EvaluationResults):
        if self.find_best_policy:
            if self.optimization_specification == SpecificationChecker.Constants.REACHABILITY:
                if tf.math.equal(evaluation_result.reach_probs[-1], tf.reduce_max(evaluation_result.reach_probs)):
                    self.save_best_policy()
            elif self.optimization_specification == SpecificationChecker.Constants.REWARD:
                if tf.math.equal(evaluation_result.returns[-1], tf.reduce_max(evaluation_result.returns)):
                    self.save_best_policy()
            else:
                raise ValueError("Unknown optimization specification")
