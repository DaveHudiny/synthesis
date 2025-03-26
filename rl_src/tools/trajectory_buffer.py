import numpy as np
from environment.environment_wrapper_vec import EnvironmentWrapperVec
from tf_agents.trajectories import Trajectory


class TrajectoryBuffer:
    class EpisodeOutcomes:
        def __init__(self, virtual_rewards: list = [], cumulative_rewards: list = [], goals_achieved: list = [], traps: list = []):
            assert type(virtual_rewards) == list and type(cumulative_rewards) == list and type(
                goals_achieved) == list and type(traps) == list, "All arguments must be lists."
            self.virtual_rewards = virtual_rewards
            self.cumulative_rewards = cumulative_rewards
            self.goals_achieved = goals_achieved
            self.traps_achieved = traps

        def add_episode_outcome(self, virtual_reward, cumulative_reward, goal_achieved, trap_achieved):
            self.virtual_rewards.append(virtual_reward)
            self.cumulative_rewards.append(cumulative_reward)
            self.goals_achieved.append(goal_achieved)
            self.traps_achieved.append(trap_achieved)

        def clear(self):
            self.virtual_rewards = []
            self.cumulative_rewards = []
            self.goals_achieved = []
            self.traps_achieved = []

    def __init__(self, environment: EnvironmentWrapperVec = None):
        self.virtual_rewards = []
        self.real_rewards = []
        self.finished = []
        self.finished_successfully = []
        self.finished_truncated = []
        self.finished_traps = []
        self.tf_step_types = []
        self.environment = environment
        self.episode_outcomes = self.EpisodeOutcomes([], [], [], [])

    def add_batched_step(self, traj: Trajectory):
        environment = self.environment
        self.virtual_rewards.append(environment.virtual_reward.numpy())
        self.real_rewards.append(environment.default_rewards.numpy())
        self.finished.append(environment.dones)
        self.finished_successfully.append(environment.goal_state_mask)
        self.finished_truncated.append(environment.truncated)
        self.finished_traps.append(environment.anti_goal_state_mask)

    def numpize_lists(self):
        self.virtual_rewards = np.array(self.virtual_rewards).T
        self.real_rewards = np.array(self.real_rewards).T
        self.finished = np.array(self.finished).T
        self.finished_successfully = np.array(self.finished_successfully).T
        self.finished_truncated = np.array(self.finished_truncated).T
        self.finished_traps = np.array(self.finished_traps).T

    def update_outcomes(self):
        self.numpize_lists()
        outcomes = self.episode_outcomes
        finished_true_indices = np.argwhere(self.finished == True)
        prev_index = np.array([0, 0])
        for index in finished_true_indices:
            if index[0] != prev_index[0]:
                prev_index = np.array([index[0], 0])
            in_episode_reward = np.sum(
                self.real_rewards[prev_index[0], prev_index[1]:index[1]+1])
            in_episode_virtual_reward = np.sum(
                self.virtual_rewards[prev_index[0], prev_index[1]:index[1]+1])
            goal_achieved = self.finished_successfully[index[0], index[1]]
            trap_achieved = self.finished_traps[index[0], index[1]]
            outcomes.add_episode_outcome(
                in_episode_virtual_reward, in_episode_reward, goal_achieved, trap_achieved)
            prev_index = index
            prev_index[1] += 1

    def final_update_of_results(self, updator: callable = None):
        self.update_outcomes()
        # print(self.episode_outcomes.cumulative_rewards)
        avg_return = np.mean(self.episode_outcomes.cumulative_rewards)
        avg_episode_return = np.mean(self.episode_outcomes.virtual_rewards)
        reach_prob = np.mean(self.episode_outcomes.goals_achieved)
        trap_prob = np.mean(self.episode_outcomes.traps_achieved)
        episode_variance = np.var(self.episode_outcomes.cumulative_rewards)
        virtual_variance = np.var(self.episode_outcomes.virtual_rewards)
        combined_variance = np.var(
            np.array(self.episode_outcomes.cumulative_rewards) + np.array(self.episode_outcomes.virtual_rewards))
        if updator:
            # updator(avg_return, avg_episode_return, reach_prob, returns, successes)
            updator(avg_return, avg_episode_return, reach_prob, episode_variance, num_episodes=len(self.episode_outcomes.cumulative_rewards),
                    trap_reach_prob=trap_prob, virtual_variance=virtual_variance, combined_variance=combined_variance)
        return avg_return, avg_episode_return, reach_prob

    def clear(self):
        self.virtual_rewards = []
        self.real_rewards = []
        self.finished = []
        self.finished_successfully = []
        self.finished_truncated = []
        self.finished_traps = []
        self.episode_outcomes.clear()
