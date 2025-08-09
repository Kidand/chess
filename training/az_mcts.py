#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AlphaZero MCTS for Xiangqi (clean redesign).

English comments per user preference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import math
import numpy as np
import torch

from backend.xiangqi import parse_fen, generate_legal_moves, other, is_in_check
from backend.encoding import board_to_planes, move_to_index, index_to_move


@dataclass
class MCTSConfig:
    num_simulations: int = 800
    cpuct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    cap_boost: float = 2.0
    check_boost: float = 1.5
    fpu_value: float = -0.2  # first play urgency value for unvisited children (from current player POV)
    c_base: float = 19652.0
    c_init: float = 1.25


class Node:
    def __init__(self, prior: float):
        self.prior = float(prior)
        self.visit = 0
        self.value_sum = 0.0
        self.children: Dict[int, Node] = {}

    def value(self) -> float:
        return 0.0 if self.visit == 0 else self.value_sum / self.visit


class AZMCTS:
    def __init__(self, net: torch.nn.Module, device: torch.device, cfg: MCTSConfig):
        self.net = net
        self.device = device
        self.cfg = cfg

    @torch.no_grad()
    def _infer(self, planes: np.ndarray) -> Tuple[np.ndarray, float]:
        x = torch.from_numpy(planes[None, ...]).to(self.device)
        use_cuda = x.is_cuda
        use_mps = (self.device.type == 'mps')
        if use_cuda:
            ac = torch.autocast(device_type='cuda', dtype=torch.float16)
        elif use_mps:
            ac = torch.autocast(device_type='mps', dtype=torch.float16)
        else:
            class _Null:
                def __enter__(self): return None
                def __exit__(self, *a): return False
            ac = _Null()
        with ac:
            p_logits, v = self.net(x)
            p = torch.softmax(p_logits, dim=-1).float().cpu().numpy()[0]
            vv = float(v.float().cpu().numpy()[0])
        return p, vv

    def run(self, fen: str, temperature: float = 1.0, apply_root_noise: bool = True) -> Tuple[np.ndarray, int, float]:
        b, side = parse_fen(fen)
        planes = board_to_planes(b, side)
        policy, v0 = self._infer(planes)

        root = Node(0.0)
        legal = generate_legal_moves(b, side)
        priors: Dict[int, float] = {}
        for m in legal:
            idx = move_to_index(m.from_row, m.from_col, m.to_row, m.to_col)
            pri = policy[idx]
            if b[m.to_row][m.to_col] != '.':
                pri *= self.cfg.cap_boost
            # check boost
            brd = [row[:] for row in b]
            brd[m.to_row][m.to_col] = brd[m.from_row][m.from_col]
            brd[m.from_row][m.from_col] = '.'
            if is_in_check(brd, other(side)):
                pri *= self.cfg.check_boost
            priors[idx] = pri
        s = sum(priors.values()) + 1e-12
        for k in list(priors.keys()):
            priors[k] /= s
        # Dirichlet noise at root
        if apply_root_noise and len(priors) > 0 and self.cfg.dirichlet_frac > 0.0:
            noise = np.random.dirichlet([self.cfg.dirichlet_alpha] * len(priors))
            keys = list(priors.keys())
            for i, k in enumerate(keys):
                priors[k] = (1 - self.cfg.dirichlet_frac) * priors[k] + self.cfg.dirichlet_frac * noise[i]
        root.children = {k: Node(v) for k, v in priors.items()}

        # Simulations
        for _ in range(self.cfg.num_simulations):
            node = root
            path: List[Tuple[Node, int, str, list]] = []
            cur_b = [row[:] for row in b]
            cur_side = side
            # Select to leaf
            while node.children:
                total = sum(ch.visit for ch in node.children.values()) + 1
                best_ucb = -1e9
                best_k = None
                best_child = None
                for k, ch in node.children.items():
                    q = ch.value() if ch.visit > 0 else self.cfg.fpu_value
                    # AlphaZero cpuct schedule
                    cpuct_eff = self.cfg.cpuct * (math.log((total + self.cfg.c_base + 1.0) / self.cfg.c_base) + self.cfg.c_init)
                    ucb = q + cpuct_eff * ch.prior * math.sqrt(total) / (1 + ch.visit)
                    if ucb > best_ucb:
                        best_ucb = ucb; best_k = k; best_child = ch
                assert best_k is not None and best_child is not None
                path.append((node, best_k, cur_side, cur_b))
                node = best_child
                fr, fc, tr, tc = index_to_move(best_k)
                cur_b = [row[:] for row in cur_b]
                cur_b[tr][tc] = cur_b[fr][fc]
                cur_b[fr][fc] = '.'
                cur_side = other(cur_side)

            # Expand
            planes2 = board_to_planes(cur_b, cur_side)
            p2, v2 = self._infer(planes2)
            legals2 = generate_legal_moves(cur_b, cur_side)
            if legals2:
                pri2: Dict[int, float] = {}
                for m in legals2:
                    idx = move_to_index(m.from_row, m.from_col, m.to_row, m.to_col)
                    val = p2[idx]
                    if cur_b[m.to_row][m.to_col] != '.':
                        val *= self.cfg.cap_boost
                    brd2 = [row[:] for row in cur_b]
                    brd2[m.to_row][m.to_col] = brd2[m.from_row][m.from_col]
                    brd2[m.from_row][m.from_col] = '.'
                    if is_in_check(brd2, other(cur_side)):
                        val *= self.cfg.check_boost
                    pri2[idx] = val
                s2 = sum(pri2.values()) + 1e-12
                for k in list(pri2.keys()): pri2[k] /= s2
                node.children = {k: Node(v) for k, v in pri2.items()}

            # Backup
            g = v2
            for n, _, _, _ in path:
                n.visit += 1
                n.value_sum += g
                g = -g

        # Extract policy from visits at root
        visits = np.zeros(8100, dtype=np.float32)
        for k, ch in root.children.items():
            visits[k] = ch.visit
        if temperature <= 1e-6:
            action = int(np.argmax(visits))
        else:
            probs = visits ** (1.0 / max(1e-3, temperature))
            s = probs.sum()
            if s > 0:
                probs /= s
                action = int(np.random.choice(np.arange(8100), p=probs))
            else:
                action = int(np.argmax(visits))
        return visits, action, v0


