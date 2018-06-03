

def calculate_return(list_of_reward, gamma):
    """
    Calculates the return for a list of
    rewards given from a rollouts
    Args:
        list_of_reward: reward rollout from environment
        gamma: discount factor
    Returns:

    """
    G = 0
    for r in reversed(list_of_reward):
        G = gamma * G + r

    return G