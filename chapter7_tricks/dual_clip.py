"""
PPO (Policy) Dual Clip.

The Dual-Clip Proximal Policy Optimization (PPO) method is designed to constrain updates to the policy,
effectively preventing it from diverging excessively from its preceding iterations. This approach thereby ensures a
more stable and reliable learning process during training. For further details, please refer to the source paper:
Mastering Complex Control in MOBA Games with Deep Reinforcement Learning. <link https://arxiv.org/pdf/1912.09729.pdf link>.
"""
import torch


def ppo_dual_clip(logp_new: torch.FloatTensor, logp_old: torch.FloatTensor, adv: torch.FloatTensor, clip_ratio: float,
                  dual_clip: float) -> torch.FloatTensor:
    """
    **Overview**:
        This function implements the Proximal Policy Optimization (PPO) policy loss with dual-clip mechanism, which is
        a variant of PPO that provides more reliable and stable training by limiting the updates to the policy, preventing
        it from deviating too much from its previous versions.
    Arguments:
        - logp_new (:obj:`torch.FloatTensor`): The log probability calculated by the new policy.
        - logp_old (:obj:`torch.FloatTensor`): The log probability calculated by the old policy.
        - adv (:obj:`torch.FloatTensor`): The advantage value, which measures how much better an action is compared to
            the average action at that state.
        - clip_ratio (:obj:`float`): The clipping ratio used to limit the change of policy during an update.
        - dual_clip (:obj:`float`): The dual clipping ratio used to further limit the change of policy during an update.
    Returns:
        - policy_loss (:obj:`torch.FloatTensor`): The calculated policy loss, which is the objective we want to minimize
            for improving the policy.
    """
    # This is the ratio of the new policy probability to the old policy probability.
    # $$r(\theta) = \frac{\pi_{new}(a|s)}{\pi_{old}(a|s)}$$
    ratio = torch.exp(logp_new - logp_old)
    # The first clipping operation is performed here, we limit the update to be within a certain range.
    # $$clip_1 = min(r(\theta)*A(s,a), clip(r(\theta), 1-clip\_ratio, 1+clip\_ratio)*A(s,a))$$
    surr1 = ratio * adv
    surr2 = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv
    clip1 = torch.min(surr1, surr2)
    # The second clipping operation is performed here, we further limit the update to be within a stricter range.
    # $$clip_2 = max(clip_1, dual\_clip * A(s,a))$$
    clip2 = torch.max(clip1, dual_clip * adv)
    # We only apply the dual-clip when the advantage is negative, i.e., when the action is worse than the average.
    policy_loss = -(torch.where(adv < 0, clip2, clip1)).mean()
    return policy_loss


# delimiter
def test_ppo_dual_clip() -> None:
    """
    **Overview**:
        This function tests the ppo_dual_clip function. It generates some sample data, calculates the policy loss
        using the ppo_dual_clip function, and checks if the returned value is a scalar.
    """
    # Generate random data for testing. The batch size is 6.
    B = 6
    logp_new = torch.randn(B)
    logp_old = torch.randn(B)
    adv = torch.randn(B)
    # Calculate policy loss using the ppo_dual_clip function.
    policy_loss = ppo_dual_clip(logp_new, logp_old, adv, 0.2, 0.2)
    # Assert that the returned policy loss is a scalar (i.e., its shape is an empty tuple).
    assert policy_loss.shape == torch.Size([])


if __name__ == "__main__":
    # Execute the test function.
    test_ppo_dual_clip()
