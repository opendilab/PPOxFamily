"""
PPO Dual Clip. These method limit the updates to policy, preventing it from deviating too much from its previous versions and ensuring more stable and reliable training.
"""
import torch


def ppo_dual_clip(logp_new: torch.FloatTensor, logp_old: torch.FloatTensor, adv: torch.FloatTensor, clip_ratio: float, dual_clip: float) -> torch.FloatTensor:
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
def test_ppo_dual_clip() -> None:
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
    policy_loss = ppo_dual_clip(logp_new, logp_old, adv, 0.2, 0.2)
    # The returned value is a scalar.
    assert policy_loss.shape == torch.Size([])


if __name__ == "__main__":
    test_ppo_dual_clip()
