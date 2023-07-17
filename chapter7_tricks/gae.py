"""
Generalized Advantage Estimator(GAE). This method estimates the advantage function by taking into account both immediate and future rewards and thus enhance the efficiency and stability of PPO.
"""


import torch


def gae(data: tuple, gamma: float = 0.99, lambda_: float = 0.97) -> torch.FloatTensor:
    """
    **Overview**:
        Implementation of Generalized Advantage Estimator (arXiv:1506.02438)
    Arguments:
        - data (:obj:`namedtuple`): gae input data with fields ['value', 'reward'], which contains some episodes or \
            trajectories data.
        - gamma (:obj:`float`): the future discount factor, should be in [0, 1], defaults to 0.99.
        - lambda (:obj:`float`): the gae parameter lambda, should be in [0, 1], defaults to 0.97, when lambda -> 0, \
            it induces bias, but when lambda -> 1, it has high variance due to the sum of terms.
    Returns:
        - adv (:obj:`torch.FloatTensor`): the calculated advantage
    Shapes:
        - value (:obj:`torch.FloatTensor`): :math:`(T, B)`, where T is trajectory length and B is batch size
        - next_value (:obj:`torch.FloatTensor`): :math:`(T, B)`
        - reward (:obj:`torch.FloatTensor`): :math:`(T, B)`
        - adv (:obj:`torch.FloatTensor`): :math:`(T, B)`
    """
    # Unpack the data.
    value, next_value, reward, done, traj_flag = data
    done = done.float()
    traj_flag = traj_flag.float()
    # Expand ``done`` for possible broadcast operation in multi-agent cases
    if len(value.shape) == 2:
        done = done.unsqueeze(1)
    # If `done = 1`, this indicates that next value should be set as `0`.
    next_value *= (1 - done)
    # $$\delta_t=-V_{\phi}(s_t)+r_t+V_{\phi}(s_{t+1})$$
    delta = reward + gamma * next_value - value
    # If `traj_flag = 0`, the factor will be set as 0. Otherwise, the factor is $$\gamma * \lambda$$
    factor = gamma * lambda_ * (1 - traj_flag)
    # Calculate `adv` in a reversed sequence. Consider the definition of GAE: $$A^{GAE}_t = \sum_{i=1}\gamma^{i-1}\lambda^{i-1}\delta_{t+i-1}$$ Rewrite the equation above in a recurrent form, we finally have: $$A^{GAE}_t = \delta_t + \gamma\lambda A^{GAE}_{t+1}$$
    adv = torch.zeros_like(value)
    gae_item = torch.zeros_like(value[0])
    for t in reversed(range(reward.shape[0])):
        gae_item = delta[t] + factor[t] * gae_item
        adv[t] = gae_item
    return adv


# delimiter
def test_gae() -> None:
    """
    **Overview**:
        Test `gae` function.
    """
    # Generate data, batch size is 2 and trajectory length is 3.
    B, T = 2, 3
    value = torch.randn(T, B)
    next_value = torch.randn(T, B)
    reward = torch.randn(T, B)
    done = None
    traj_flag = None
    data = (value, next_value, reward, done, traj_flag)
    # Calculate GAE values.
    gae_value = gae(data)
    assert gae_value.shape == (T, B)


if __name__ == "__main__":
    test_gae()
