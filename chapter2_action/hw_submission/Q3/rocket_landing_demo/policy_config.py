exp_config = {
    'type': 'ppo',
    'on_policy': True,
    'cuda': False,
    'action_space': 'discrete',
    'discount_factor': 0.99,
    'gae_lambda': 0.95,
    'epoch_per_collect': 5,
    'batch_size': 64,
    'learning_rate': 0.0003,
    'weight_decay': 0,
    'value_weight': 0.5,
    'entropy_weight': 0.01,
    'clip_ratio': 0.2,
    'adv_norm': False,
    'value_norm': True,
    'ppo_param_init': True,
    'grad_norm': 0.5,
    'n_sample': 2048,
    'unroll_len': 1,
    'deterministic_eval': True,
    'model': {
        'encoder_hidden_size_list': [64, 64, 128],
        'actor_head_hidden_size': 128,
        'critic_head_hidden_size': 128
    },
    'cfg_type': 'PPOFPolicyDict'
}
