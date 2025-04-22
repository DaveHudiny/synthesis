
import tensorflow as tf

from rl_src.interpreters.extracted_fsc.table_based_policy import TableBasedPolicy
from rl_src.environment.environment_wrapper_vec import EnvironmentWrapperVec

from paynt.quotient.fsc import FSC
from paynt.quotient.pomdp import PomdpQuotient

class ConstructorFSC:
    """Class to construct a Finite State Controller (FSC) from different policies, e.g. TableBasedPolicy.
    This class provides a method to convert a table-based policy into an FSC, which can be used for reinforcement learning tasks.
    """

    @staticmethod
    def __create_action_function(tf_action_function : tf.Tensor):
        """Creates the action function for the FSC.
        Args:
            tf_action_function (tf.Tensor): The action function to be used in the FSC.
        Returns:
            tf.Tensor: The created action function.
        """
        return tf.cast(tf_action_function, dtype=tf.int32).numpy().tolist()
    
    @staticmethod
    def __create_update_function(tf_update_function : tf.Tensor):
        """Creates the update function for the FSC.
        Args:
            tf_update_function (tf.Tensor): The update function to be used in the FSC.
        Returns:
            tf.Tensor: The created update function.
        """
        return tf.cast(tf_update_function, dtype=tf.int32).numpy().tolist()
    
    @staticmethod
    def __create_observation_labels(pomdp_quotient : PomdpQuotient):
        return pomdp_quotient.observation_labels
    
    @staticmethod
    def __create_action_labels(environment : EnvironmentWrapperVec):
        return environment.action_keywords

    @staticmethod
    def construct_fsc_from_table_based_policy(
        table_based_policy: TableBasedPolicy,
        environment_wrapper: EnvironmentWrapperVec,
        pomdp_quotient: PomdpQuotient
        ) -> FSC:
        """Constructs a Finite State Controller (FSC) from a table-based policy.
        Args:
            table_based_policy (TableBasedPolicy): The table-based policy to be converted into an FSC.
            environment_wrapper (EnvironmentWrapperVec): The environment wrapper used to create the FSC.
        Returns:
            FSC: The constructed FSC.
        """
        # Create a new FSC object
        action_function = ConstructorFSC.__create_action_function(table_based_policy.tf_observation_to_action_table)
        update_function = ConstructorFSC.__create_update_function(table_based_policy.tf_observation_to_update_table)
        observation_labels = ConstructorFSC.__create_observation_labels(pomdp_quotient)
        action_labels = ConstructorFSC.__create_action_labels(environment_wrapper)
        
        num_observations = len(observation_labels)
        num_nodes = len(action_function)

        # Create the FSC
        fsc = FSC(
            action_function=action_function,
            update_function=update_function,
            observation_labels=observation_labels,
            action_labels=action_labels,
            is_deterministic=True,
            num_observations=num_observations,
            num_nodes=num_nodes,
        )
        return fsc