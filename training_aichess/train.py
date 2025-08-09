#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aichess-style trainer wrapper that consumes buffer and trains.

English comments per user preference.
"""

from __future__ import annotations

import os
import pickle
import time
from tqdm import tqdm

from ..aichess.collect import CollectPipeline
from ..aichess.collect import zip_array
from ..aichess.config import CONFIG
from ..aichess.pytorch_net import PolicyValueNet


def run_train(buffer_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    net = PolicyValueNet()
    lr_multiplier = 1.0
    kl_targ = CONFIG['kl_targ']
    batch_size = CONFIG['batch_size']
    epochs = CONFIG['epochs']
    bar = tqdm(desc='Train', dynamic_ncols=True)
    while True:
        # load buffer snapshot
        try:
            with open(os.path.join(buffer_path, 'train_data_buffer.pkl'), 'rb') as f:
                data_file = pickle.load(f)
            data_buffer = data_file['data_buffer']
            iters = data_file['iters']
        except Exception:
            time.sleep(5); continue

        if len(data_buffer) < batch_size:
            time.sleep(5); continue

        # sample minibatch
        import random
        mini_batch = random.sample(data_buffer, batch_size)
        mini_batch = [zip_array.recovery_state_mcts_prob(d) for d in mini_batch]
        state_batch = [d[0] for d in mini_batch]
        mcts_probs_batch = [d[1] for d in mini_batch]
        winner_batch = [d[2] for d in mini_batch]

        # old probs/values for KL
        old_probs, old_v = net.policy_value(state_batch)

        last_loss = 0.0
        for _ in range(epochs):
            loss, entropy = net.train_step(state_batch, mcts_probs_batch, winner_batch, lr=1e-3 * lr_multiplier)
            new_probs, new_v = net.policy_value(state_batch)
            import numpy as np
            kl = (old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10))).sum(axis=1).mean()
            if kl > kl_targ * 4: break
            last_loss = float(loss)

        # adapt lr
        if kl > kl_targ * 2 and lr_multiplier > 0.1:
            lr_multiplier /= 1.5
        elif kl < kl_targ / 2 and lr_multiplier < 10:
            lr_multiplier *= 1.5

        # save
        out_file = os.path.join(out_dir, f'policy_iter_{iters}.pt')
        net.save_model(out_file)
        bar.set_postfix_str(f"iters={iters} kl={kl:.4f} loss={last_loss:.4f} lr_mul={lr_multiplier:.2f} buf={len(data_buffer)}")
        bar.update(1)
        time.sleep(1)


if __name__ == '__main__':
    run_train('runs/aichess_buffer', 'runs/aichess_models')


