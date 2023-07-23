"""
The Generalized Advantage Estimator (GAE) is a technique that accurately estimates the advantage function by considering both immediate and future rewards.
This approach not only improves the efficacy of Proximal Policy Optimization (PPO), but also enhances its stability.
You can find more detailed information in this paper <link https://arxiv.org/pdf/1506.02438.pdf link>.
"""


import torch


def gae(data: tuple, gamma: float = 0.99, lambda_: float = 0.97) -> torch.FloatTensor:
    """
    **Overview**:
        Implementation of the Generalized Advantage Estimator (GAE) as proposed in arXiv:1506.02438.
        This function calculates the advantages, which are used to update policy parameters in reinforcement learning.

    Arguments:
        - data (:obj:`namedtuple`): Tuple containing trajectory data including state values, next state values, rewards, done flags, and trajectory flags.
            Please note that the 'done' flag signals the termination of an episode, whereas the 'traj_flag' indicates the completion of a trajectory,
            which represents a segment within an episode.
        - gamma (:obj:`float`): Discount factor for future rewards, should be in the range [0, 1]. Default is 0.99.
        - lambda_ (:obj:`float`): The decay rate for the GAE, should be in the range [0, 1]. Default is 0.97.
            As lambda approaches 0, it introduces bias, and as lambda approaches 1, it increases variance due to the cumulative effect of terms.
    Returns:
        - adv (:obj:`torch.FloatTensor`): The calculated advantage estimates.
    Shapes:
        - value (:obj:`torch.FloatTensor`): Size of (T, B), where T is the length of the trajectory and B is the batch size.
        - next_value (:obj:`torch.FloatTensor`):  Size of (T, B)
        - reward (:obj:`torch.FloatTensor`): Size of (T, B)
        - adv (:obj:`torch.FloatTensor`): Size of (T, B)
    """
    # Unpack the input data.
    value, next_value, reward, done, traj_flag = data

    # Convert the done and trajectory flags to tensor format.
    done = torch.tensor(done).float()
    traj_flag = torch.tensor(traj_flag).float()

    # If done equals 1, it indicates the end of an episode, thus the next state value should be 0.
    next_value *= (1 - done)

    # Calculate the temporal difference (TD) error for each time step.
    # $$\delta_t=-V_{\phi}(s_t)+r_t+V_{\phi}(s_{t+1})$$
    delta = reward + gamma * next_value - value

    # Set the GAE decay factor. If traj_flag equals 1, the factor will be 0. Otherwise, the factor is gamma * lambda.
    factor = gamma * lambda_ * (1 - traj_flag)

    # Calculate ``adv`` in a reversed sequence.
    # Consider the definition of GAE: $$A^{GAE}_t = \sum_{i=1}\gamma^{i-1}\lambda^{i-1}\delta_{t+i-1}$$
    # Rewrite the equation above in a recurrent form, we finally have: $$A^{GAE}_t = \delta_t + \gamma\lambda A^{GAE}_{t+1}$$

    # Initialize the advantage tensor.
    adv = torch.zeros_like(value)
    # Calculate the advantage for each time step in reverse order.
    gae_item = torch.zeros_like(value[0])
    for t in reversed(range(reward.shape[0])):
        gae_item = delta[t] + factor[t] * gae_item
        adv[t] = gae_item

    # Return the calculated advantage estimates.
    return adv


# delimiter
def test_gae() -> None:
    """
    **Overview**:
        Test the GAE function with randomly generated data.
    """
    # Generate random data with trajectory length 10 and batch size 5.
    T, B = 10, 5
    value = torch.randn(T, B)
    next_value = torch.randn(T, B)
    reward = torch.randn(T, B)
    done = torch.randint(0, 2, (T, B)).to(torch.bool)  # Generate random boolean tensor for done flags.
    traj_flag = torch.randint(0, 2, (T, B)).to(torch.bool)  # Generate random boolean tensor for trajectory flags.
    data = (value, next_value, reward, done, traj_flag)

    # Calculate GAE values.
    gae_value = gae(data)

    # Assert that the calculated GAE values have the correct shape.
    assert gae_value.shape == (T, B)


if __name__ == "__main__":
    # Execute the test function.
    test_gae()
