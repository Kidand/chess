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
from backend.policy_planes import map_move_to_plane_id, policy_index_from_move


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
    # Capture prior shaping: 'uniform' uses cap_boost for any capture; 'tiered' scales by captured piece type
    cap_mode: str = "uniform"  # 'uniform' | 'tiered'
    cap_tier: Optional[Dict[str, float]] = None  # e.g., {'R':2.0,'C':1.6,'H':1.5,'S':1.2,'A':1.1,'E':1.1}
    # Linear blend to 1.0: effective = 1.0 + (tier_value - 1.0) * cap_tier_scale  (0->1.0, 1->tier)
    cap_tier_scale: float = 1.0


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

    def _is_structured_policy(self) -> bool:
        return getattr(self.net, "policy_head_type", "flat") == "structured"

    def _cap_multiplier(self, board, move) -> float:
        """Return prior multiplier for captures based on config.

        - 'uniform': cap_boost if destination is occupied.
        - 'tiered': multiply based on captured piece importance with linear scale.
        """
        tr, tc = move.to_row, move.to_col
        cap = board[tr][tc]
        if cap == '.':
            return 1.0
        if self.cfg.cap_mode == 'tiered':
            tier = self.cfg.cap_tier or {}
            key = cap.upper()
            base = float(tier.get(key, 1.2))  # default small boost if unspecified
            scale = float(max(0.0, min(1.0, self.cfg.cap_tier_scale)))
            return 1.0 + (base - 1.0) * scale
        # uniform
        return float(self.cfg.cap_boost)

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
            # Read prior from policy head (flat or structured)
            if self._is_structured_policy():
                plane_id = map_move_to_plane_id(b, side, m.from_row, m.from_col, m.to_row, m.to_col)
                if plane_id is None:
                    pri = 1e-9
                else:
                    pol_idx = policy_index_from_move(plane_id, m.from_row, m.from_col)
                    pri = float(policy[pol_idx])
            else:
                idx = move_to_index(m.from_row, m.from_col, m.to_row, m.to_col)
                pri = float(policy[idx])
            # Capture prior shaping
            pri *= self._cap_multiplier(b, m)
            # check boost
            brd = [row[:] for row in b]
            brd[m.to_row][m.to_col] = brd[m.from_row][m.from_col]
            brd[m.from_row][m.from_col] = '.'
            if is_in_check(brd, other(side)):
                pri *= self.cfg.check_boost
            k_idx = move_to_index(m.from_row, m.from_col, m.to_row, m.to_col)
            priors[k_idx] = pri
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
            path: List[Tuple[Node, int, str, list]] = []  # (parent, child_key, side_to_move_at_parent, board_before_move)
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
                    if self._is_structured_policy():
                        plane_id = map_move_to_plane_id(cur_b, cur_side, m.from_row, m.from_col, m.to_row, m.to_col)
                        if plane_id is None:
                            val = 1e-9
                        else:
                            pol_idx = policy_index_from_move(plane_id, m.from_row, m.from_col)
                            val = float(p2[pol_idx])
                    else:
                        idx = move_to_index(m.from_row, m.from_col, m.to_row, m.to_col)
                        val = float(p2[idx])
                    # Capture prior shaping at expansion
                    val *= self._cap_multiplier(cur_b, m)
                    brd2 = [row[:] for row in cur_b]
                    brd2[m.to_row][m.to_col] = brd2[m.from_row][m.from_col]
                    brd2[m.from_row][m.from_col] = '.'
                    if is_in_check(brd2, other(cur_side)):
                        val *= self.cfg.check_boost
                    k_idx = move_to_index(m.from_row, m.from_col, m.to_row, m.to_col)
                    pri2[k_idx] = val
                s2 = sum(pri2.values()) + 1e-12
                for k in list(pri2.keys()): pri2[k] /= s2
                node.children = {k: Node(v) for k, v in pri2.items()}

            # Backup along edges (parent->child), updating child nodes' visits
            g = v2  # value from the perspective of side to move at leaf
            for parent, child_k, _, _ in path:
                child = parent.children[child_k]
                child.visit += 1
                child.value_sum += g
                g = -g

        # Extract policy from visits at root
        visits = np.zeros(8100, dtype=np.float32)
        for k, ch in root.children.items():
            visits[k] = ch.visit
        if visits.sum() <= 0:
            # Fallback to normalized priors if visits failed to accumulate (shouldn't happen after fix)
            prior_sum = sum(ch.prior for ch in root.children.values()) + 1e-12
            for k, ch in root.children.items():
                visits[k] = ch.prior / prior_sum
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


