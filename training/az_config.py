#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AlphaZero training configuration for Xiangqi.

English comments per user preference.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AZConfig:
    # Model
    channels: int = 256
    blocks: int = 12

    # Self-play / MCTS
    num_simulations: int = 1600
    cpuct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    temperature_moves: int = 20  # number of opening moves with T=1, afterwards T=0
    resign_threshold: float = -0.95  # auto-resign if value below for consecutive plies
    resign_consec: int = 3
    no_capture_draw_plies: int = 60  # early draw if no capture for many plies
    mcts_batch: int = 512

    # Parallelism
    envs_per_rank: int = 8

    # Training
    batch_size: int = 2048
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs_per_segment: int = 1
    save_every_segments: int = 1

    # Replay buffer
    replay_capacity: int = 500_000
    replay_warmup: int = 50_000

    # Paths
    out_dir: str = "runs/xq"


