import pytest
import numpy as np
from sheep_env import SheepEnv


@pytest.mark.unittest
def test_naive():
    env = SheepEnv(level=10)
    obs = env.reset()
    print(env.observation_space)
    assert isinstance(env.scene[0].to_json(), str)

    while True:
        action_mask = obs['action_mask']
        action = np.random.choice(len(action_mask), p=action_mask / action_mask.sum())
        obs, rew, done, info = env.step(action)
        print(env.bucket, rew, done)
        assert isinstance(obs, dict)
        assert set(obs.keys()) == set(['item_obs', 'bucket_obs', 'global_obs', 'action_mask'])
        assert np.isscalar(rew)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        if done:
            break
