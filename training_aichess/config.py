CONFIG = {
    'kill_action': 30,            # 和棋回合数（aichess 默认）
    'dirichlet': 0.2,            # Dirichlet alpha
    'play_out': 1200,            # 每步模拟次数
    'c_puct': 5.0,               # PUCT 权重
    'buffer_size': 100000,       # 经验池大小
    'pytorch_model_path': 'current_policy.pt',
    'train_data_buffer_path': 'train_data_buffer.pkl',
    'batch_size': 512,           # 训练 batch
    'kl_targ': 0.02,             # KL 目标
    'epochs': 5,                 # 每次更新步数
    'game_batch_num': 3000,      # 训练总步数
    'train_update_interval': 600,# 训练更新间隔秒
    'use_redis': False,
}

