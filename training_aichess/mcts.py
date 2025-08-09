#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
from training.az_mcts import MCTSConfig, AZMCTS
from backend.xiangqi import parse_fen, to_fen, other, generate_legal_moves
from backend.encoding import board_to_planes, move_to_index


class BoardAdapter:
    def __init__(self, board, side):
        self.board = board
        self.side = side

    def availables(self):
        moves = generate_legal_moves(self.board, self.side)
        idxs = [move_to_index(m.from_row, m.from_col, m.to_row, m.to_col) for m in moves]
        return idxs


class MCTSPlayer:
    def __init__(self, policy_value_fn, c_puct=5, n_playout=1200, is_selfplay=1):
        self.cfg = MCTSConfig(num_simulations=n_playout, cpuct=c_puct)
        self.policy_value_fn = policy_value_fn
        self.is_selfplay = is_selfplay

    def get_action(self, b, side, temp=1.0):
        mcts = AZMCTS(self.policy_value_fn.__self__.policy_value_net, self.policy_value_fn.__self__.device, self.cfg)
        fen = to_fen(b, side)
        visits, action, _ = mcts.run(fen, temperature=temp)
        return action, visits


