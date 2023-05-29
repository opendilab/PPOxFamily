"""
This document mainly includes three tricks to boost the performance of PPO:
- Generalized Advantage Estimator(GAE). This method estimates the advantage function by taking into account both immediate and future rewards and thus enhance the efficiency and stability of PPO.
- Value Clip and Dual Clip. These two methods respectively limit the updates to the value function and policy, preventing them from deviating too much from their previous versions and ensuring more stable and reliable training.
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
    if done is None:
        done = torch.zeros_like(reward, device=reward.device)
    if traj_flag is None:
        traj_flag = done
    done = done.float()
    traj_flag = traj_flag.float()
    # For some marl case: value is shaped (T, B, A), reward is shaped (T, B). Then unsqueeze them into 3-dimensional tensors.
    if len(value.shape) == len(reward.shape) + 1:  
        reward = reward.unsqueeze(-1)
        done = done.unsqueeze(-1)
        traj_flag = traj_flag.unsqueeze(-1)
    # If `done = 1`, this indicates that next value should be set as `0`.
    next_value *= (1 - done)
    # $$\delta_t=-V_{\phi}(s_t)+r_t+V_{\phi}(s_{t+1})$$
    delta = reward + gamma * next_value - value
    # If `traj_flag = 0`, the factor will be set as 0. Otherwise, the factor is $$\gamma * \lambda$$
    factor = gamma * lambda_ * (1 - traj_flag)
    # Calculate `adv` in a reversed sequence. This is because $$A^{GAE}_t = \delta_t + \gamma\lambda A^{GAE}_{t+1}$$
    adv = torch.zeros_like(value)
    gae_item = torch.zeros_like(value[0])
    for t in reversed(range(reward.shape[0])):
        gae_item = delta[t] + factor[t] * gae_item
        adv[t] = gae_item
    return adv


# delimiter
def value_clip(value_old: torch.FloatTensor, value_new: torch.FloatTensor, return_: torch.FloatTensor, clip_ratio: float = 0.2) -> torch.FloatTensor:
    """
    **Overview**:
        Implementation of Value Clip.
    Arguments:
        - value_old (:obj:`torch.FloatTensor`): Value calculated by old policy.
        - value_new (:obj:`torch.FloatTensor`): Value calculated by new policy.
        - return_ (:obj:`torch.FloatTensor`): The return value (target).
        - clip_ratio (:obj:`float`): The clip ratio of value. Default is 0.2.
    Returns:
        - value_loss (:obj:`torch.FloatTensor`): the calculated value loss.
    """
    # $$V_{clip} = V_{old} + clip(V_{new} - V_{old}, -clip\_ratio, clip\_ratio)$$
    value_clip = value_old + (value_new - value_old).clamp(-clip_ratio, clip_ratio)
    # $$V_1 = (return - V_{new})^2$$
    v1 = (return_ - value_new).pow(2)
    # $$V_2 = (return - V_{clip})^2$$
    v2 = (return_ - value_clip).pow(2)
    # $$loss = 0.5 * weight * max(V_1, V_2)$$
    value_loss = 0.5 * (torch.max(v1, v2)).mean()
    return value_loss


# delimiter
def dual_clip(logp_new: torch.FloatTensor, logp_old: torch.FloatTensor, adv: torch.FloatTensor, clip_ratio: float, dual_clip: float) -> torch.FloatTensor:
    """
    **Overview**:
        Implementation of Dual Clip.
    Arguments:
        - logp_new (:obj:`torch.FloatTensor`): log_p calculated by old policy.
        - logp_old (:obj:`torch.FloatTensor`): log_p calculated by new policy.
        - adv (:obj:`torch.FloatTensor`): The advantage value.
        - clip_ratio (:obj:`float`): The clip ratio of policy.
        - dual_clip (:obj:`float`): The dual clip ratio of policy.
    Returns:
        - policy_loss (:obj:`torch.FloatTensor`): the calculated policy loss.
    """
    # $$r(\theta) = \frac{\pi_{new}(a|s)}{\pi_{old}(a|s)}$$
    ratio = torch.exp(logp_new - logp_old)
    # $$clip_1 = min(r(\theta)*A(s,a), clip(r(\theta), 1-clip\_ratio, 1+clip\_ratio)*A(s,a))$$
    surr1 = ratio * adv
    surr2 = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv
    clip1 = torch.min(surr1, surr2)
    # $$clip_2 = max(clip_1, dual\_clip * A(s,a))$$
    clip2 = torch.max(clip1, dual_clip * adv)
    # Only use dual_clip when adv < 0.
    policy_loss = -(torch.where(adv < 0, clip2, clip1)).mean()
    return policy_loss


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


# delimiter
def test_value_clip() -> None:
    """
    **Overview**:
        Test `value_clip` function.
    """
    # Generate data, batch size is 6.
    B = 6
    value_old = torch.randn(B)
    value_new = torch.randn(B)
    return_ = torch.randn(B)
    # Calculate value loss with value clip.
    value_loss = value_clip(value_old, value_new, return_)
    # The returned value is a scalar.
    assert value_loss.shape == torch.Size([])


# delimiter
def test_dual_clip() -> None:
    """
    **Overview**:
        Test `dual_clip` function.
    """
    # Generate data, batch size is 6.
    B = 6
    logp_new = torch.randn(B)
    logp_old = torch.randn(B)
    adv = torch.randn(B)
    # Calculate policy loss with policy loss.
    policy_loss = dual_clip(logp_new, logp_old, adv, 0.2, 0.2)
    # The returned value is a scalar.
    assert policy_loss.shape == torch.Size([])


if __name__ == "__main__":
    test_gae()
    test_value_clip()
    test_dual_clip()
