import tensorflow as tf

def compute_log_prob_loss(log_probs, weights=None):
    """
    Computes the log probability loss for a given set of log probabilities and actions.

    Args:
        log_probs (tf.Tensor): Log probabilities of the actions.
        actions (tf.Tensor): Actions taken by the agent.
        weights (tf.Tensor, optional): Weights for the loss. Defaults to None.

    Returns:
        tf.Tensor: The computed log probability loss.
    """
    if weights is None:
        weights = tf.ones_like(log_probs, dtype=tf.float32)
    
    # Compute the loss
    loss = -tf.reduce_sum(log_probs * weights)
    
    return loss
