#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-play data collector (aichess-style) for Xiangqi.

English comments per user preference.
"""

from __future__ import annotations

import os
import pickle
import time
from collections import deque
from typing import Deque, List

from tqdm import tqdm

from ..aichess.collect import CollectPipeline as _ACollect


def run_collect(output_path: str):
    cp = _ACollect()
    bar = tqdm(desc='Collect', dynamic_ncols=True)
    while True:
        iters = cp.collect_selfplay_data(n_games=1)
        # Persist buffer snapshot under new folder
        data = {'data_buffer': cp.data_buffer, 'iters': cp.iters}
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, 'train_data_buffer.pkl'), 'wb') as f:
            pickle.dump(data, f)
        bar.set_postfix_str(f"iters={iters} size={len(cp.data_buffer)}")
        bar.update(1)
        time.sleep(1)


if __name__ == '__main__':
    run_collect('runs/aichess_buffer')


