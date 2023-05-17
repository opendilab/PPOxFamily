exp_config = {
    'env': {
        'manager': {
            'episode_num': float("inf"),
            'max_retry': 5,
            'step_timeout': None,
            'auto_reset': True,
            'reset_timeout': None,
            'retry_type': 'reset',
            'retry_waiting_time': 0.1,
            'shared_memory': True,
            'copy_on_get': True,
            'context': 'fork',
            'wait_num': float("inf"),
            'step_wait_timeout': None,
            'connect_timeout': 60,
            'reset_inplace': False,
            'cfg_type': 'SyncSubprocessEnvManagerDict',
            'type': 'subprocess'
        },
        'stop_value': 6000,
        'type': 'mujoco_multi',
        'import_names': ['dizoo.multiagent_mujoco.envs.multi_mujoco_env'],
        'scenario': 'Ant-v2',
        'agent_conf': '2x4d',
        'agent_obsk': 2,
        'add_agent_id': False,
        'episode_limit': 1000,
        'collector_env_num': 8,
        'evaluator_env_num': 8,
        'n_evaluator_episode': 8
    },
    'policy': {
        'model': {
            'agent_num': 2,
            'agent_obs_shape': 54,
            'global_obs_shape': 111,
            'action_shape': 4,
            'action_space': 'continuous'
        },
        'learn': {
            'learner': {
                'train_iterations': 1000000000,
                'dataloader': {
                    'num_workers': 0
                },
                'log_policy': True,
                'hook': {
                    'load_ckpt_before_run': '',
                    'log_show_after_iter': 100,
                    'save_ckpt_after_iter': 10000,
                    'save_ckpt_after_run': True
                },
                'cfg_type': 'BaseLearnerDict'
            },
            'epoch_per_collect': 3,
            'batch_size': 800,
            'learning_rate': 0.0005,
            'value_weight': 0.5,
            'entropy_weight': 0.001,
            'clip_ratio': 0.2,
            'adv_norm': True,
            'value_norm': True,
            'ppo_param_init': True,
            'grad_clip_type': 'clip_norm',
            'grad_clip_value': 5,
            'ignore_done': False
        },
        'collect': {
            'collector': {
                'deepcopy_obs': False,
                'transform_obs': False,
                'collect_print_freq': 100,
                'cfg_type': 'SampleSerialCollectorDict',
                'type': 'sample'
            },
            'unroll_len': 1,
            'discount_factor': 0.99,
            'gae_lambda': 0.95,
            'env_num': 8,
            'n_sample': 3200
        },
        'eval': {
            'evaluator': {
                'eval_freq': 1000,
                'render': {
                    'render_freq': -1,
                    'mode': 'train_iter'
                },
                'cfg_type': 'InteractionSerialEvaluatorDict',
                'n_episode': 8,
                'stop_value': 6000
            },
            'env_num': 8
        },
        'other': {
            'replay_buffer': {
                'type': 'advanced',
                'replay_buffer_size': 4096,
                'max_use': float("inf"),
                'max_staleness': float("inf"),
                'alpha': 0.6,
                'beta': 0.4,
                'anneal_step': 100000,
                'enable_track_used_data': False,
                'deepcopy': False,
                'thruput_controller': {
                    'push_sample_rate_limit': {
                        'max': float("inf"),
                        'min': 0
                    },
                    'window_seconds': 30,
                    'sample_min_limit_ratio': 1
                },
                'monitor': {
                    'sampled_data_attr': {
                        'average_range': 5,
                        'print_freq': 200
                    },
                    'periodic_thruput': {
                        'seconds': 60
                    }
                },
                'cfg_type': 'AdvancedReplayBufferDict'
            },
            'commander': {
                'cfg_type': 'BaseSerialCommanderDict'
            }
        },
        'on_policy': True,
        'cuda': True,
        'multi_gpu': False,
        'bp_update_sync': True,
        'traj_len_inf': False,
        'type': 'ppo_command',
        'priority': False,
        'priority_IS_weight': False,
        'recompute_adv': True,
        'action_space': 'continuous',
        'nstep_return': False,
        'multi_agent': True,
        'transition_with_policy_data': True,
        'cfg_type': 'PPOCommandModePolicyDict'
    },
    'exp_name': 'multi_mujoco_ant_2x4_ppo',
    'seed': 0
}
