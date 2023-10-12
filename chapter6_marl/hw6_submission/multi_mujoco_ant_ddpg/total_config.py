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
            'twin_critic': False,
            'agent_obs_shape': 54,
            'global_obs_shape': 111,
            'action_shape': 4,
            'action_space': 'regression',
            'actor_head_hidden_size': 256,
            'critic_head_hidden_size': 256
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
            'update_per_collect': 10,
            'batch_size': 256,
            'learning_rate_actor': 0.001,
            'learning_rate_critic': 0.001,
            'ignore_done': False,
            'target_theta': 0.005,
            'discount_factor': 0.99,
            'actor_update_freq': 1,
            'noise': False
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
            'noise_sigma': 0.1,
            'n_sample': 400
        },
        'eval': {
            'evaluator': {
                'eval_freq': 500,
                'render': {
                    'render_freq': -1,
                    'mode': 'train_iter'
                },
                'cfg_type': 'InteractionSerialEvaluatorDict',
                'n_episode': 8,
                'stop_value': 6000
            }
        },
        'other': {
            'replay_buffer': {
                'type': 'naive',
                'replay_buffer_size': 100000,
                'deepcopy': False,
                'enable_track_used_data': False,
                'periodic_thruput_seconds': 60,
                'cfg_type': 'NaiveReplayBufferDict'
            },
            'commander': {
                'cfg_type': 'BaseSerialCommanderDict'
            }
        },
        'type': 'ddpg_command',
        'cuda': True,
        'on_policy': False,
        'priority': False,
        'priority_IS_weight': False,
        'random_collect_size': 0,
        'transition_with_policy_data': False,
        'action_space': 'continuous',
        'reward_batch_norm': False,
        'multi_agent': True,
        'cfg_type': 'DDPGCommandModePolicyDict'
    },
    'exp_name': 'multi_mujoco_ant_2x4_ddpg',
    'seed': 0
}
