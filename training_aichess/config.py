CONFIG = {
    'kill_action': 60,
    'dirichlet': 0.2,
    'play_out': 1200,
    'c_puct': 5.0,
    'buffer_size': 200000,
    'pytorch_model_path': 'current_policy.pt',
    'train_data_buffer_path': 'train_data_buffer.pkl',
    'batch_size': 1024,
    'kl_targ': 0.02,
    'epochs': 5,
    'game_batch_num': 3000,
    'train_update_interval': 300,
    'use_redis': False,
}

