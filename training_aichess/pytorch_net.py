#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from training.az_model import XQAZNet
from backend.encoding import board_to_planes


class PolicyValueNet:
    def __init__(self, model_file=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.policy_value_net = XQAZNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_value_net.parameters(), lr=1e-3, weight_decay=2e-3)
        if model_file:
            try:
                ckpt = torch.load(model_file, map_location=self.device)
                sd = ckpt.get('state_dict', ckpt)
                self.policy_value_net.load_state_dict(sd, strict=False)
            except Exception:
                pass

    def policy_value(self, state_batch):
        # state_batch: (N, 15,10,9)
        self.policy_value_net.eval()
        x = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            with autocast(enabled=self.device.type=='cuda'):
                log_act_probs, value = self.policy_value_net(x)
        log_act_probs = log_act_probs.float().cpu().numpy()
        value = value.float().cpu().numpy()
        act_probs = np.exp(log_act_probs)
        return act_probs, value

    def policy_value_fn(self, board):
        # board is current engine board: we convert to planes
        planes = board_to_planes(board, 'r')  # side not used here for aichess adapter
        x = planes[None, ...]
        probs, v = self.policy_value(x)
        probs = probs[0]
        # legal mask由外部mcts提供；这里直接返回全量分布
        legal_positions = [(i, probs[i]) for i in range(len(probs)) if probs[i] > 1e-12]
        return legal_positions, v

    def save_model(self, model_file):
        torch.save({'state_dict': self.policy_value_net.state_dict()}, model_file)

    def train_step(self, state_batch, mcts_probs, winner_batch, lr=1e-3):
        self.policy_value_net.train()
        x = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
        p = torch.tensor(mcts_probs, dtype=torch.float32, device=self.device)
        v = torch.tensor(winner_batch, dtype=torch.float32, device=self.device)
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        self.optimizer.zero_grad()
        log_act_probs, value = self.policy_value_net(x)
        value_loss = F.mse_loss(value, v.squeeze(-1))
        policy_loss = -torch.mean(torch.sum(p * log_act_probs, dim=1))
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1)).item()
        return float(loss.item()), float(entropy)


