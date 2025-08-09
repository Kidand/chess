#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, os, pickle, random
from collections import deque
import numpy as np
from tqdm import tqdm

from training_aichess.config import CONFIG
from training_aichess.pytorch_net import PolicyValueNet


class TrainPipeline:
    def __init__(self, init_model=None):
        self.learn_rate = 1e-3
        self.lr_multiplier = 1.0
        self.batch_size = CONFIG['batch_size']
        self.epochs = CONFIG['epochs']
        self.kl_targ = CONFIG['kl_targ']
        self.check_freq = 100
        self.game_batch_num = CONFIG['game_batch_num']
        self.buffer_size = CONFIG['buffer_size']
        self.data_buffer = deque(maxlen=self.buffer_size)
        if init_model and os.path.exists(init_model):
            self.policy_value_net = PolicyValueNet(model_file=init_model)
        else:
            self.policy_value_net = PolicyValueNet()

    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = np.array([d[0] for d in mini_batch]).astype('float32')
        mcts_probs_batch = np.array([d[1] for d in mini_batch]).astype('float32')
        winner_batch = np.array([d[2] for d in mini_batch]).astype('float32')
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(state_batch, mcts_probs_batch, winner_batch, self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs+1e-10) - np.log(new_probs+1e-10)), axis=1))
            if kl > self.kl_targ * 4: break
        if kl > self.kl_targ*2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ/2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        return loss, entropy, kl

    def run(self):
        pbar = tqdm(total=self.game_batch_num, desc='Train', dynamic_ncols=True)
        for i in range(self.game_batch_num):
            # load latest buffer
            while True:
                try:
                    with open(CONFIG['train_data_buffer_path'], 'rb') as f:
                        data = pickle.load(f)
                        self.data_buffer = data['data_buffer']
                        iters = data['iters']
                    break
                except Exception:
                    time.sleep(5)
            if len(self.data_buffer) > self.batch_size:
                loss, entropy, kl = self.policy_update()
                self.policy_value_net.save_model(CONFIG['pytorch_model_path'])
                # periodic ckpt
                if (i+1) % CONFIG['check_freq'] == 0:
                    tag = f"models/current_policy_batch{i+1}.pt"
                    os.makedirs('models', exist_ok=True)
                    self.policy_value_net.save_model(tag)
                pbar.set_postfix_str(f"iters={iters} loss={loss:.4f} ent={entropy:.4f} kl={kl:.5f} lr_mul={self.lr_multiplier:.3f} buf={len(self.data_buffer)}")
                pbar.update(1)
            time.sleep(CONFIG['train_update_interval'])
        pbar.close()


if __name__=='__main__':
    TrainPipeline(init_model=CONFIG['pytorch_model_path']).run()


