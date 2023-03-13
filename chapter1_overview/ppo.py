"""
PyTorch implementation of Proximal Policy Optimization (PPO)
PPO is one of the most common algorithms in reinforcement learning, which combines Actor-Critic methods and Trust Region Policy Optimization.
For the policy part, PPO combines clipped optimization target and pessimistic bound to update policy. For the value function part, PPO usually uses classical temporal difference methods (such as GAE).
This final target function is formulated as:
$$\min(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_k}(a_{t}|s_{t})}A^{\theta_k}(s_{t},a_{t}),\text{clip}(\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_k}(a_{t}|s_{t})}, 1-\epsilon,1+\epsilon)A^{\theta_k}(s_{t},a_{t}))$$
This document mainly includes:
- Implementation of PPO error.
- Main function (test function)
"""
from typing import Optional, Tuple
from collections import namedtuple
import torch
import numpy as np

ppo_policy_data = namedtuple('ppo_policy_data', ['logit_new', 'logit_old', 'action', 'adv', 'weight'])
ppo_policy_loss = namedtuple('ppo_policy_loss', ['policy_loss', 'entropy_loss'])
ppo_info = namedtuple('ppo_info', ['approx_kl', 'clipfrac'])


def ppo_policy_error(data: namedtuple,
                     clip_ratio: float = 0.2,
                     dual_clip: Optional[float] = None) -> Tuple[namedtuple, namedtuple]:
    """
    **Overview**:
        Implementation of Proximal Policy Optimization (PPO) <link https://arxiv.org/pdf/1707.06347.pdf link> with entropy bonus, value_clip and dual_clip.
    """
    # Unpack data: $$<\pi_{new}(a|s), \pi_{old}(a|s), a, A^{\pi_{old}}(s, a), w>$$
    logit_new, logit_old, action, adv, weight = data
    # Prepare weight for default cases.
    if weight is None:
        weight = torch.ones_like(adv)
    # Prepare policy distribution from logit and get log propability.
    dist_new = torch.distributions.categorical.Categorical(logits=logit_new)
    dist_old = torch.distributions.categorical.Categorical(logits=logit_old)
    logp_new = dist_new.log_prob(action)
    logp_old = dist_old.log_prob(action)
    # Entropy bonus: $$\frac 1 N \sum_{n=1}^{N} \sum_{a^n}\pi_{new}(a^n|s^n) log(\pi_{new}(a^n|s^n))$$
    # P.S. the final loss is ``policy_loss - entropy_weight * entropy_loss`` .
    dist_new_entropy = dist_new.entropy()
    entropy_loss = (dist_new_entropy * weight).mean()
    # Importance sampling weight: $$r(\theta) = \frac{\pi_{new}(a|s)}{\pi_{old}(a|s)}$$
    ratio = torch.exp(logp_new - logp_old)
    # Original surrogate objective: $$r(\theta) A^{\pi_{old}}(s, a)$$
    surr1 = ratio * adv
    # <b>Clipped surrogate objective:</b> $$clip(r(\theta), 1-\epsilon, 1+\epsilon) A^{\pi_{old}}(s, a)$$
    surr2 = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv
    # Dual clip proposed by <link https://arxiv.org/abs/1912.09729 link> .
    # Only use dual_clip when adv < 0.
    if dual_clip is not None:
        clip1 = torch.min(surr1, surr2)
        clip2 = torch.max(clip1, dual_clip * adv)
        policy_loss = -(torch.where(adv < 0, clip2, clip1) * weight).mean()
    # PPO-Clipped Loss: $$min(r(\theta) A^{\pi_{old}}(s, a), clip(r(\theta), 1-\epsilon, 1+\epsilon) A^{\pi_{old}}(s, a))$$
    # Multiply sample-wise weight and reduce mean in batch dimension.
    else:
        policy_loss = (-torch.min(surr1, surr2) * weight).mean()
    # Add some visualization metrics to monitor optimization status.
    with torch.no_grad():
        approx_kl = (logp_old - logp_new).mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped).float().mean().item()
    # Return final loss items and information.
    return ppo_policy_loss(policy_loss, entropy_loss), ppo_info(approx_kl, clipfrac)


# delimiter
def test_ppo(clip_ratio, dual_clip):
    """
    **Overview**:
        Test function of PPO, for both forward and backward operations.
    """
    # batch size=4, action=32
    B, N = 4, 32
    # Generate logit_new, logit_old, action, adv.
    logit_new = torch.randn(B, N).requires_grad_(True)
    logit_old = logit_new + torch.rand_like(logit_new) * 0.1
    action = torch.randint(0, N, size=(B, ))
    adv = torch.rand(B)
    data = ppo_policy_data(logit_new, logit_old, action, adv, None)
    # Compute PPO error.
    loss, info = ppo_policy_error(data, clip_ratio=clip_ratio, dual_clip=dual_clip)
    # Assert the loss is differentiable.
    assert all([np.isscalar(i) for i in info])
    assert logit_new.grad is None
    total_loss = sum(loss)
    total_loss.backward()
    assert isinstance(logit_new.grad, torch.Tensor)


if __name__ == '__main__':
    test_ppo(0.2, 0.5)
