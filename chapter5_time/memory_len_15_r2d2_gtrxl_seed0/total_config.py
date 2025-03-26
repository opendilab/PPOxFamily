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
            'shared_memory': False,
            'copy_on_get': True,
            'context': 'fork',
            'wait_num': float("inf"),
            'step_wait_timeout': None,
            'connect_timeout': 60,
            'reset_inplace': False,
            'cfg_type': 'SyncSubprocessEnvManagerDict',
            'type': 'subprocess'
        },
        'stop_value': 1.0,
        'type': 'bsuite',
        'import_names': ['dizoo.bsuite.envs.bsuite_env'],
        'collector_env_num': 8,
        'evaluator_env_num': 1,
        'n_evaluator_episode': 20,
        'env_id': 'memory_len/15'
    },
    'policy': {
        'model': {
            'obs_shape': 3,
            'action_shape': 2,
            'memory_len': 0,
            'hidden_size': 64,
            'gru_bias': 1.0
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
            'multi_gpu': False,
            'update_per_collect': 8,
            'batch_size': 64,
            'learning_rate': 0.0005,
            'target_update_theta': 0.001,
            'ignore_done': False,
            'value_rescale': False,
            'init_memory': 'zero'
        },
        'collect': {
            'collector': {
                'deepcopy_obs': False,
                'transform_obs': False,
                'collect_print_freq': 100,
                'cfg_type': 'SampleSerialCollectorDict',
                'type': 'sample'
            },
            'each_iter_n_sample': 32,
            'env_num': 8
        },
        'eval': {
            'evaluator': {
                'eval_freq': 10,
                'render': {
                    'render_freq': -1,
                    'mode': 'train_iter'
                },
                'cfg_type': 'InteractionSerialEvaluatorDict',
                'n_episode': 20,
                'stop_value': 1.0
            },
            'env_num': 1
        },
        'other': {
            'replay_buffer': {
                'type': 'advanced',
                'replay_buffer_size': 50000,
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
            'eps': {
                'type': 'exp',
                'start': 0.95,
                'end': 0.05,
                'decay': 100000.0
            },
            'commander': {
                'cfg_type': 'BaseSerialCommanderDict'
            }
        },
        'type': 'r2d2_gtrxl_command',
        'cuda': True,
        'on_policy': False,
        'priority': True,
        'priority_IS_weight': True,
        'discount_factor': 0.997,
        'nstep': 3,
        'burnin_step': 0,
        'unroll_len': 35,
        'seq_len': 35,
        'cfg_type': 'R2D2GTrXLCommandModePolicyDict'
    },
    'exp_name': 'memory_len_15_r2d2_gtrxl_seed0',
    'seed': 0
}
