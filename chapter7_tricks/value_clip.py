"""
PPO Value Clip. This method limit the updates to the value function, preventing the value for a certain state changes too fast.
"""
import torch


def ppo_value_clip(value_old: torch.FloatTensor, value_new: torch.FloatTensor, return_: torch.FloatTensor, clip_ratio: float = 0.2) -> torch.FloatTensor:
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
def test_ppo_value_clip() -> None:
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
    value_loss = ppo_value_clip(value_old, value_new, return_)
    # The returned value is a scalar.
    assert value_loss.shape == torch.Size([])


if __name__ == "__main__":
    test_ppo_value_clip()
