import pytest
import torch
from sheep_model import SheepModel


@pytest.mark.unittest
def test_naive():
    B, M = 3, 30
    model = SheepModel(item_num=30, item_encoder_type='two_stage_MLP')
    data = {
        'item_obs': torch.randn(B, M, 60),
        'bucket_obs': torch.randn(B, 30),
        'global_obs': torch.randn(B, 17),
        'action_mask': torch.randint(0, 2, size=(M, )).bool()
    }

    output = model.forward(data, mode='compute_actor_critic')
    assert output['logit'].shape == (B, M)
    assert output['value'].shape == (B, )
