import numpy as np
from compact_rl.rl.environment.environment_wrapper_vec import EnvironmentWrapperVec
from tf_agents.trajectories import Trajectory


class TrajectoryBuffer:
    class EpisodeOutcomes:
        def __init__(self, virtual_rewards: list = [], cumulative_rewards: list = [], goals_achieved: list = [], 
                     traps: list = [], discounted_reward: float = [], bad_outcomes: list = [],
                     discounted_reachability: list = [], discounted_return: list = []):
            assert type(virtual_rewards) == list and type(cumulative_rewards) == list and type(
                goals_achieved) == list and type(traps) == list, "All arguments must be lists."
            self.virtual_rewards = virtual_rewards
            self.cumulative_rewards = cumulative_rewards
            self.goals_achieved = goals_achieved
            self.traps_achieved = traps
            self.discounted_virt_rewards = discounted_reward
            self.bad_outcomes = bad_outcomes
            self.discounted_reachabilities = discounted_reachability
            self.discounted_returns = discounted_return

        def add_episode_outcome(self, virtual_reward, cumulative_reward, goal_achieved, trap_achieved, discounted_reward, bad_outcome, discounted_reachability, discounted_return):
            self.virtual_rewards.append(virtual_reward)
            self.cumulative_rewards.append(cumulative_reward)
            self.goals_achieved.append(goal_achieved)
            self.traps_achieved.append(trap_achieved)
            self.discounted_virt_rewards.append(discounted_reward)
            self.bad_outcomes.append(bad_outcome)
            self.discounted_reachabilities.append(discounted_reachability)
            self.discounted_returns.append(discounted_return)

        def clear(self):
            self.virtual_rewards = []
            self.cumulative_rewards = []
            self.goals_achieved = []
            self.traps_achieved = []
            self.discounted_virt_rewards = []
            self.bad_outcomes = []

    def __init__(self, environment: EnvironmentWrapperVec = None, truncation_point: int = None):
        self.virtual_rewards = []
        self.real_rewards = []
        self.discounts = []
        self.finished = []
        self.finished_successfully = []
        self.finished_truncated = []
        self.finished_traps = []
        self.tf_step_types = []
        self.is_in_bad_state = []
        self.environment = environment
        self.truncation_point = truncation_point if truncation_point is not None else (environment.args.max_steps if environment else None)
        self.episode_outcomes = self.EpisodeOutcomes([], [], [], [], [], [])
        self.average_episode_length = 0
        self.counted_episodes = 0

    def add_batched_step(self, traj: Trajectory):
        """WARNING: This function uses the current state of the environment to add the step. 
        If the environment is desynchronized with the trajectory, it will not work properly."""
        environment = self.environment
        self.virtual_rewards.append(traj.reward.numpy())
        self.real_rewards.append(environment.orig_reward.numpy())
        self.finished.append(environment.dones)
        self.finished_successfully.append(environment.goal_state_mask)
        self.finished_truncated.append(environment.truncated)
        self.finished_traps.append(environment.anti_goal_state_mask)
        self.is_in_bad_state.append(environment.is_in_bad_state)
        self.discounts.append(traj.discount.numpy())

    def numpize_lists(self):
        self.virtual_rewards = np.array(self.virtual_rewards).T
        self.real_rewards = np.array(self.real_rewards).T
        self.finished = np.array(self.finished).T
        self.finished_successfully = np.array(self.finished_successfully).T
        self.finished_truncated = np.array(self.finished_truncated).T
        self.finished_traps = np.array(self.finished_traps).T
        self.discounts = np.array(self.discounts).T
        self.is_in_bad_state = np.array(self.is_in_bad_state).T

    def update_outcomes(self, truncate : bool = True):
        self.numpize_lists()
        outcomes = self.episode_outcomes
        finished_true_indices = np.argwhere(self.finished == True)
        prev_index = np.array([0, 0])
        cumulative_length = 0
        counted_episodes = 0
        number_of_sampled_steps = self.finished.shape[1]
        for index in finished_true_indices:
            if index[0] != prev_index[0]:
                prev_index = np.array([index[0], 0])
            if truncate and number_of_sampled_steps - prev_index[1] < self.truncation_point:
                continue
            in_episode_reward = np.sum(
                self.real_rewards[prev_index[0], prev_index[1]:index[1]+1])
            in_episode_virtual_reward = np.sum(
                self.virtual_rewards[prev_index[0], prev_index[1]:index[1]+1])
            goal_achieved = self.finished_successfully[index[0], index[1]]
            trap_achieved = self.finished_traps[index[0], index[1]]
            discount_cumprod = np.cumprod(
                self.discounts[prev_index[0], prev_index[1]:index[1]+1])
            discounted_virt_reward = np.sum(
                self.virtual_rewards[prev_index[0], prev_index[1]:index[1]+1] * discount_cumprod)
            discounted_return = np.sum(
                self.real_rewards[prev_index[0], prev_index[1]:index[1]+1] * discount_cumprod)
            discounted_reachability = goal_achieved * discount_cumprod[-1] if goal_achieved else 0.0
            in_episode_bad_state = np.any(
                self.is_in_bad_state[prev_index[0], prev_index[1]:index[1]+1])

            outcomes.add_episode_outcome(
                in_episode_virtual_reward, in_episode_reward, goal_achieved, trap_achieved, discounted_virt_reward, in_episode_bad_state, discounted_reachability, discounted_return)
            cumulative_length += index[1] - prev_index[1]
            prev_index = index
            prev_index[1] += 1
            counted_episodes += 1
        self.average_episode_length = cumulative_length / counted_episodes
        self.counted_episodes = counted_episodes

    def final_update_of_results(self, updator: callable = None, truncate: bool = True):
        self.update_outcomes(truncate=truncate)
        # print(self.episode_outcomes.cumulative_rewards)
        avg_return = np.mean(self.episode_outcomes.cumulative_rewards)
        avg_episode_return = np.mean(self.episode_outcomes.virtual_rewards)
        reach_prob = np.mean(self.episode_outcomes.goals_achieved)
        trap_prob = np.mean(self.episode_outcomes.traps_achieved)
        episode_variance = np.var(self.episode_outcomes.cumulative_rewards)
        virtual_variance = np.var(self.episode_outcomes.virtual_rewards)
        combined_variance = np.var(
            np.array(self.episode_outcomes.cumulative_rewards) + np.array(self.episode_outcomes.virtual_rewards))
        avg_discounted_virt_reward = np.mean(self.episode_outcomes.discounted_virt_rewards)
        avg_bad_outcome_prob = np.mean(self.episode_outcomes.bad_outcomes)
        discounted_return = np.mean(self.episode_outcomes.discounted_returns)
        discounted_reachability = np.mean(self.episode_outcomes.discounted_reachabilities)
        if updator:
            # updator(avg_return, avg_episode_return, reach_prob, returns, successes)
            updator(avg_return, avg_episode_return, reach_prob, episode_variance, 
                    num_episodes=len(self.episode_outcomes.cumulative_rewards),
                    trap_reach_prob=trap_prob, virtual_variance=virtual_variance, 
                    combined_variance=combined_variance, 
                    average_episode_length=self.average_episode_length,
                    counted_episodes=self.counted_episodes,
                    discounted_virtual_rewards = avg_discounted_virt_reward,
                    average_bad_outcome_prob=avg_bad_outcome_prob,
                    discounted_return = discounted_return,
                    discounted_reachability = discounted_reachability,
            )
        return avg_return, avg_episode_return, reach_prob

    def clear(self):
        self.virtual_rewards = []
        self.real_rewards = []
        self.finished = []
        self.finished_successfully = []
        self.finished_truncated = []
        self.finished_traps = []
        self.episode_outcomes.clear()
        self.average_episode_length = 0
        self.counted_episodes = 0
        self.discounts = []
        self.is_in_bad_state = []
        self.discounted_reachabilities = []
        self.discounted_returns = []
