"""
PyTorch implementation of Policy Gradient (PG)
Policy gradient (also known as REINFORCE) is a classical method for learning a policy.
Each $$(s_t,a_t)$$ will be used to compute corresponding log probability. Then the probability is back-propagated then and calculate gradient. The gradient will be multiplied  by a weight value, which is the accumulated return in this game.
The final target function is formulated as:
$$- \frac 1 N \sum_{n=1}^{N} log(\pi(a^n|s^n)) G_t^n$$
This document mainly includes:
- Implementation of PG error.
- Main function (test function)
"""
from collections import namedtuple
import torch

pg_data = namedtuple('pg_data', ['logit', 'action', 'return_'])
pg_loss = namedtuple('pg_loss', ['policy_loss', 'entropy_loss'])


def pg_error(data: namedtuple) -> namedtuple:
    """
    **Overview**:
        Implementation of PG (Policy Gradient)
    """
    # Unpack data: $$<\pi(a|s), a, G_t>$$
    logit, action, return_ = data
    # Prepare policy distribution from logit and get log propability.
    dist = torch.distributions.categorical.Categorical(logits=logit)
    log_prob = dist.log_prob(action)
    # Policy loss: $$- \frac 1 N \sum_{n=1}^{N} log(\pi(a^n|s^n)) G_t^n$$
    policy_loss = -(log_prob * return_).mean()
    # Entropy bonus: $$\frac 1 N \sum_{n=1}^{N} \sum_{a^n}\pi(a^n|s^n) log(\pi(a^n|s^n))$$
    # P.S. the final loss is ``policy_loss - entropy_weight * entropy_loss`` .
    entropy_loss = dist.entropy().mean()
    # Return the concrete loss items.
    return pg_loss(policy_loss, entropy_loss)


# delimiter
def test_pg():
    """
    **Overview**:
        Test function of PG, for both forward and backward operations.
    """
    # batch size=4, action=32
    B, N = 4, 32
    # Generate logit, action, return_.
    logit = torch.randn(B, N).requires_grad_(True)
    action = torch.randint(0, N, size=(B, ))
    return_ = torch.randn(B) * 2
    # Compute PG error.
    data = pg_data(logit, action, return_)
    loss = pg_error(data)
    # Assert the loss is differentiable.
    assert all([l.shape == tuple() for l in loss])
    assert logit.grad is None
    total_loss = sum(loss)
    total_loss.backward()
    assert isinstance(logit.grad, torch.Tensor)


if __name__ == '__main__':
    test_pg()
