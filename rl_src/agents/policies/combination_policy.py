from tf_agents.policies import TFPolicy

from tf_agents.trajectories.policy_step import PolicyStep

import enum

import tensorflow as tf

class CombinationSettings(enum.IntEnum):
    """Enum for combination settings."""
    PRIMARY_POLICY = 0
    ALL_POLICIES = 1

class CombinationPolicy(TFPolicy):
    def __init__(self, policies: list[TFPolicy], time_step_spec, action_spec, name=None,
                 observation_and_action_constraint_splitter: callable = None, enable_masking=False,
                 combination_settings: CombinationSettings = CombinationSettings.PRIMARY_POLICY,
                 primary_policy_index: bool = 0):
        """Policy for combining more than one policy in one larger policy. Currently implemented version with one dominant policy, which selects the action and the other policies are used for the policy state and info.
        
        Args:
            policies (list[TFPolicy]): List of policies to combine.
            time_step_spec (TimeStep): Time step specification.
            action_spec (TensorSpec): Action specification.
            name (str, optional): Name of the policy. Defaults to None.
            observation_and_action_constraint_splitter (callable, optional): Splitter for observation and action constraint. Defaults to None.
            enable_masking (bool, optional): Enable masking. Defaults to False.
            combination_settings (CombinationSettings, optional): Combination settings, which defines, how the action is selected. Defaults to CombinationSettings.PRIMARY_POLICY.
            primary_policy_index (bool, optional): Index of the primary policy. Defaults to 0.
        """
        self.policies = policies
        self.enable_masking = enable_masking
        self.last_policy = None

        # Check if all policies have the same time step spec
        # for policy in policies:
        #     if policy.time_step_spec != time_step_spec:
        #         raise ValueError("All policies must have the same time step spec.")
            
        # Check if all policies have the same action 
        # for policy in policies:
        #     if policy.action_spec != action_spec:
        #         raise ValueError("All policies must have the same action spec.")

        # Create shared policy state spec
        counter = 0
        policy_state_specs = {}
        for policy in policies:
            counter += 1
            if policy.policy_state_spec == ():
                continue
            self.last_policy_state_state_counter = counter
            policy_state_specs[f"policy_{counter}"] = policy.policy_state_spec
            

        if len(policy_state_specs.keys()) <= 1:
            self.single_policy_state = True
            policy_state_specs = policy_state_specs[f"policy_{self.last_policy_state_state_counter}"]
        else:
            self.single_policy_state = False

        info_specs = {}
        counter = 0
        for policy in policies:
            counter += 1
            if policy.info_spec == ():
                continue
            self.last_policy_counter_info = counter
            info_specs[f"policy_{counter}"] = policy.info_spec
            
        if len(info_specs.keys()) <= 1:
            self.single_info = True
            info_specs = info_specs[f"policy_{self.last_policy_counter_info}"]
        else:
            self.single_info = False

        super(CombinationPolicy, self).__init__(time_step_spec, action_spec, name=name,
                                                observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
                                                policy_state_spec=policy_state_specs, info_spec=info_specs)
        
        self.combination_settings = combination_settings
        self.primary_policy_index = primary_policy_index

    def _get_initial_state(self, batch_size):
        if isinstance(self.policy_state_spec, dict):
            return {key: policy._get_initial_state(batch_size) for key, policy in zip(self.policy_state_spec.keys(), self.policies)}
        else:
            return self.last_policy._get_initial_state(batch_size)

    def _distribution(self, time_step, policy_state, seed):
        raise NotImplementedError(
            "CombinationPolicy does not support distributions")

    @tf.function
    def _action(self, time_step, policy_state, seed):
        if self.enable_masking:
            raise NotImplementedError(
                "CombinationPolicy does not support action masking")
        policy_steps = {}
        counter = 0
        for policy in self.policies:
            counter += 1
            policy_steps[f"policy_{counter}"] = policy.action(time_step, policy_state[f"policy_{counter}"], seed)
            

        selected_action = None
        if self.combination_settings == CombinationSettings.PRIMARY_POLICY:
            selected_action = policy_steps[f"policy_{self.primary_policy_index + 1}"].action
        elif self.combination_settings == CombinationSettings.ALL_POLICIES:
            raise NotImplementedError(
                "CombinationPolicy currently does not support all policies.")
        
        new_policy_state = {}
        if not self.single_policy_state:
            for key in policy_steps.keys():
                new_policy_state[key] = policy_steps[key].state
        else:
            new_policy_state = policy_steps[f"policy_{self.last_policy_state_state_counter}"].state

        new_policy_info = ()
        if not self.single_info:
            for key in policy_steps.keys():
                new_policy_info[key] = policy_steps[key].info
        else:
            new_policy_info = policy_steps[f"policy_{self.last_policy_counter_info}"].info

        return PolicyStep(selected_action, new_policy_state, new_policy_info)


