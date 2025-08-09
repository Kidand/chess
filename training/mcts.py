#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AlphaZero MCTS for Xiangqi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math
import numpy as np
import torch

from backend.xiangqi import parse_fen, generate_legal_moves, other, is_in_check
from backend.encoding import board_to_planes, move_to_index, index_to_move
from .eval_pool import BatchedEvaluator


@dataclass
class MCTSConfig:
    num_simulations: int = 1600
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    batch_size: int = 64
    cap_boost: float = 2.0  # multiply priors of capture moves
    check_boost: float = 1.5  # multiply priors of checking moves


class Node:
    def __init__(self, prior: float):
        self.prior = float(prior)
        self.visit = 0
        self.value_sum = 0.0
        self.children: Dict[int, Node] = {}

    def value(self) -> float:
        return 0.0 if self.visit == 0 else (self.value_sum / self.visit)


class MCTS:
    def __init__(self, net, device: torch.device, config: MCTSConfig, evaluator: BatchedEvaluator | None = None):
        self.net = net
        self.device = device
        self.config = config
        self.evaluator = evaluator or BatchedEvaluator(self.net, self.device, max_batch=config.batch_size)

    def infer(self, planes: np.ndarray) -> Tuple[np.ndarray, float]:
        fut = self.evaluator.submit(planes)
        p, v = fut.result()
        return p, v

    def run(self, fen: str, temperature: float = 1.0) -> Tuple[np.ndarray, int, float]:
        b, side = parse_fen(fen)
        planes = board_to_planes(b, side)
        policy, v0 = self.infer(planes)

        root = Node(0.0)
        legal = generate_legal_moves(b, side)
        priors = {}
        for m in legal:
            idx = move_to_index(m.from_row, m.from_col, m.to_row, m.to_col)
            pri = policy[idx]
            # capture boost at root
            if b[m.to_row][m.to_col] != '.':
                pri *= self.config.cap_boost
            # checking boost at root
            # simulate move
            brd = [row[:] for row in b]
            brd[m.to_row][m.to_col] = brd[m.from_row][m.from_col]
            brd[m.from_row][m.from_col] = '.'
            opp = other(side)
            if is_in_check(brd, opp):
                pri *= self.config.check_boost
            priors[idx] = pri
        s = sum(priors.values()) + 1e-8
        for k in list(priors.keys()):
            priors[k] /= s
        # Dirichlet noise at root
        if len(priors) > 0:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(priors))
            keys = list(priors.keys())
            for i, k in enumerate(keys):
                priors[k] = (1 - self.config.dirichlet_frac) * priors[k] + self.config.dirichlet_frac * noise[i]
        root.children = {k: Node(v) for k, v in priors.items()}

        # Batched selection-expansion-backup
        sims_done = 0
        while sims_done < self.config.num_simulations:
            batch = min(self.config.batch_size, self.config.num_simulations - sims_done)
            leaf_nodes = []
            leaf_paths = []
            leaf_boards = []
            leaf_sides = []
            terminal_values = []  # if terminal at selection, store value directly

            for _ in range(batch):
                node = root
                path = []
                cur_b = [row[:] for row in b]
                cur_side = side
                while node.children:
                    best_k, best_child, best_ucb = None, None, -1e9
                    total_visit = sum(ch.visit for ch in node.children.values()) + 1
                    for k, ch in node.children.items():
                        ucb = ch.value() + self.config.c_puct * ch.prior * math.sqrt(total_visit) / (1 + ch.visit)
                        if ucb > best_ucb:
                            best_ucb = ucb
                            best_k, best_child = k, ch
                    path.append((node, best_k))
                    node = best_child
                    fr, fc, tr, tc = index_to_move(best_k)
                    cur_b[tr][tc] = cur_b[fr][fc]
                    cur_b[fr][fc] = '.'
                    cur_side = other(cur_side)

                # Check terminal at leaf
                legals = generate_legal_moves(cur_b, cur_side)
                if not legals:
                    # losing for side to move -> value from root's perspective is +1
                    terminal_values.append(1.0)
                    leaf_nodes.append(node)
                    leaf_paths.append(path)
                    leaf_boards.append(None)
                    leaf_sides.append(None)
                else:
                    terminal_values.append(None)
                    leaf_nodes.append(node)
                    leaf_paths.append(path)
                    leaf_boards.append(cur_b)
                    leaf_sides.append(cur_side)

            # Inference for non-terminal leaves
            if any(lb is not None for lb in leaf_boards):
                planes_batch = [board_to_planes(lb, ls) for lb, ls in zip(leaf_boards, leaf_sides) if lb is not None]
                x = np.stack(planes_batch, axis=0)
                # batched forward with autocast
                with torch.no_grad():
                    xx = torch.from_numpy(x).to(self.device)
                    use_cuda = xx.is_cuda
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
                        p_logits, v_t = self.net(xx)
                        p_probs = torch.softmax(p_logits, dim=-1).float().cpu().numpy()
                        v_vals = v_t.float().cpu().numpy()
                # Assign back
                pi_iter = iter(p_probs)
                vv_iter = iter(v_vals)
            else:
                pi_iter = iter([])
                vv_iter = iter([])

            for i in range(batch):
                node = leaf_nodes[i]
                path = leaf_paths[i]
                if leaf_boards[i] is None:
                    v = terminal_values[i]
                else:
                    p = next(pi_iter)
                    v = float(next(vv_iter))
                    # expand children with priors over legal moves
                    cur_b = leaf_boards[i]
                    cur_side = leaf_sides[i]
                pri = {}
                legals = generate_legal_moves(cur_b, cur_side)
                for m in legals:
                    idx = move_to_index(m.from_row, m.from_col, m.to_row, m.to_col)
                    val = p[idx]
                    if cur_b[m.to_row][m.to_col] != '.':
                        val *= self.config.cap_boost
                    # checking boost
                    brd2 = [row[:] for row in cur_b]
                    brd2[m.to_row][m.to_col] = brd2[m.from_row][m.from_col]
                    brd2[m.from_row][m.from_col] = '.'
                    opp2 = other(cur_side)
                    if is_in_check(brd2, opp2):
                        val *= self.config.check_boost
                    pri[idx] = val
                    s2 = sum(pri.values()) + 1e-8
                    for k in list(pri.keys()):
                        pri[k] /= s2
                    node.children = {k: Node(vv) for k, vv in pri.items()}

                # backup
                for n, _ in path:
                    n.visit += 1
                    n.value_sum += v

            sims_done += batch

        visits = np.zeros(8100, dtype=np.float32)
        for k, ch in root.children.items():
            visits[k] = ch.visit
        if temperature == 0:
            action = int(np.argmax(visits))
        else:
            probs = visits ** (1.0 / max(1e-3, temperature))
            s = probs.sum()
            if s > 0:
                probs /= s
                action = int(np.random.choice(np.arange(8100), p=probs))
            else:
                action = int(np.argmax(visits))
        return visits, action, float(v0)


