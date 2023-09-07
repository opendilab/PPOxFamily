"""
PPO Value Clip.

The Value-Clip Proximal Policy Optimization (PPO) technique is employed to place constraints on updates to the value function,
averting rapid fluctuations in the estimated value of a given state.
This method is devised to enhance the stability and reliability of the learning process during the training phase.
For additional details, please refer to the source paper: Implementation Matters in Deep RL: A Case Study on PPO and TRPO. <link https://arxiv.org/abs/2005.12729 link>.
"""
import torch


def ppo_value_clip(value_old: torch.FloatTensor, value_new: torch.FloatTensor, return_: torch.FloatTensor,
                   clip_ratio: float = 0.2) -> torch.FloatTensor:
    """
    **Overview**:
        Implementation of Value Clip method used in PPO. The core idea is to prevent the value function from updating too rapidly for a certain state.
        This is achieved by clipping the new value within a certain range of the old value.
    Arguments:
        - value_old (:obj:`torch.FloatTensor`): The old value, calculated using the old policy.
        - value_new (:obj:`torch.FloatTensor`): The new value, calculated using the new policy.
        - return_ (:obj:`torch.FloatTensor`): The expected return value (target value).
        - clip_ratio (:obj:`float`): The clipping range for the new value, expressed as a ratio of the old value. Default is 0.2.
    Returns:
        - value_loss (:obj:`torch.FloatTensor`): The calculated value loss, represents the difference between the new and old value function.

    **Algorithm**:
        The algorithm calculates the clipped value function and then calculates two types of value losses: one between the return and the new value function,
        and the other between the return and the clipped value function. The final value loss is the average of the maximum of these two types of value losses.
    """
    # Calculate the clipped value function, which is the old value plus the difference between the new and old value, clamped within the clip ratio.
    # $$V_{clip} = V_{old} + clip(V_{new} - V_{old}, -clip\_ratio, clip\_ratio)$$
    value_clip = value_old + (value_new - value_old).clamp(-clip_ratio, clip_ratio)
    # Calculate the first type of value loss: the squared difference between the return and the new value function.
    # $$V_1 = (return - V_{new})^2$$
    v1 = (return_ - value_new).pow(2)
    # Calculate the second type of value loss: the squared difference between the return and the clipped value function.
    # $$V_2 = (return - V_{clip})^2$$
    v2 = (return_ - value_clip).pow(2)
    # Calculate the final value loss as the average of the maximum of the two types of value losses.
    # $$loss = 0.5 * weight * max(V_1, V_2)$$
    value_loss = 0.5 * (torch.max(v1, v2)).mean()
    return value_loss


# delimiter
def test_ppo_value_clip() -> None:
    """
    **Overview**:
        Test function for ppo_value_clip function. The test case generates random data and uses it to calculate the value loss.
        Then it checks whether the shape of the returned value loss is a scalar, as expected.
    """
    # Generate random data for testing. The batch size is 6.
    B = 6
    value_old = torch.randn(B)
    value_new = torch.randn(B)
    return_ = torch.randn(B)
    # Calculate the value loss using the ppo_value_clip function.
    value_loss = ppo_value_clip(value_old, value_new, return_)
    # Assert that the returned value loss is a scalar (i.e., its shape is an empty tuple).
    assert value_loss.shape == torch.Size([])


if __name__ == "__main__":
    # Execute the test function.
    test_ppo_value_clip()
