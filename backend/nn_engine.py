#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Neural-network-based engine with MCTS for Xiangqi.

English comments per user preference.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .xiangqi import parse_fen
from .encoding import board_to_planes
from training.az_model import XQAZNet
from training.az_mcts import AZMCTS, MCTSConfig


class NNMCTS:
    def __init__(self, net: nn.Module, device: torch.device, config: MCTSConfig):
        self.net = net
        self.device = device
        self.config = config

    @torch.no_grad()
    def infer(self, planes: np.ndarray) -> Tuple[np.ndarray, float]:
        x = torch.from_numpy(planes[None, ...]).to(self.device)
        use_cuda = self.device.type == 'cuda'
        use_mps = self.device.type == 'mps'
        # autocast for CUDA/MPS; on CPU keep fp32
        if use_cuda:
            autocast_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
        elif use_mps:
            autocast_ctx = torch.autocast(device_type='mps', dtype=torch.float16)
        else:
            class _Null:
                def __enter__(self):
                    return None
                def __exit__(self, *args):
                    return False
            autocast_ctx = _Null()
        with autocast_ctx:
            p_logits, v = self.net(x)
        p = torch.softmax(p_logits, dim=-1).cpu().numpy()[0]
        return p, float(v.item())

    def run(self, fen: str) -> Tuple[Dict[str, int], Dict[str, Any]]:
        # Use AZ MCTS for move selection
        mcts = AZMCTS(self.net, self.device, MCTSConfig(num_simulations=800))
        visits, action, v0 = mcts.run(fen, temperature=0.0)
        frfc = int(action // 90); trtc = int(action % 90)
        fr, fc = divmod(frfc, 9)
        tr, tc = divmod(trtc, 9)
        move = {"fr": fr, "fc": fc, "tr": tr, "tc": tc}
        meta: Dict[str, Any] = {
            "nodes": int(float(visits.sum())),
            "score": float(v0),
        }
        return move, meta


def load_model(model_path: Optional[str]) -> nn.Module:
    # Load AZ network architecture consistent with training/az_model.py
    net = XQAZNet()
    if torch.cuda.is_available():
        dev = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        dev = torch.device('mps')
    else:
        dev = torch.device('cpu')
    net.to(dev).eval()
    if model_path and os.path.isfile(model_path):
        ckpt = torch.load(model_path, map_location=dev)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            net.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            net.load_state_dict(ckpt, strict=False)
    return net


def best_move_nn(fen: str, model_path: Optional[str]):
    net = load_model(model_path)
    dev = next(net.parameters()).device
    mcts = NNMCTS(net, dev, MCTSConfig())
    return mcts.run(fen)


