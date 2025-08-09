#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bridge to run inference using aichess project's network/MCTS.

English comments per user preference.
"""

from __future__ import annotations

from typing import Dict, Tuple

from .xiangqi import parse_fen


def _piece_to_cn(p: str) -> str:
    # Map our piece chars to aichess Chinese strings
    if p == '.':
        return '一一'
    red = p.isupper()
    base = p.upper()
    name_map = {
        'R': '车',
        'H': '马',
        'E': '象',
        'A': '士',
        'K': '帅',
        'C': '炮',
        'S': '兵',
    }
    side = '红' if red else '黑'
    role = name_map.get(base, '?')
    if role == '?':
        return '一一'
    return side + role


def _fen_to_aichess_board(fen: str):
    # lazy import aichess modules
    from aichess.game import Board
    b, side = parse_fen(fen)
    board = Board()
    # Build state_list with vertical flip because our red is bottom, aichess red is top
    state_list = [['一一' for _ in range(9)] for _ in range(10)]
    for r in range(10):
        for c in range(9):
            pr = b[r][c]
            cn = _piece_to_cn(pr)
            ar = 9 - r
            state_list[ar][c] = cn
    board.state_list = state_list
    # state_deque: replicate last 4
    from collections import deque
    board.state_deque = deque(maxlen=4)
    for _ in range(4):
        board.state_deque.append([row[:] for row in state_list])
    # set current player
    if side == 'r':
        board.current_player_color = '红'
        board.current_player_id = 1
    else:
        board.current_player_color = '黑'
        board.current_player_id = 2
    board.last_move = -1
    board.kill_action = 0
    board.game_start = False
    board.action_count = 0
    return board, side


def _aichess_action_to_move_dict(act_id: int) -> Dict[str, int]:
    # Convert aichess move id -> our move dict {fr,fc,tr,tc}
    from aichess.game import move_id2move_action
    s = move_id2move_action[act_id]
    ay, ax, by, bx = int(s[0]), int(s[1]), int(s[2]), int(s[3])
    # Convert from aichess coords (top=0) back to ours (top=0 but our red was bottom so we flipped earlier):
    fr = 9 - ay
    fc = ax
    tr = 9 - by
    tc = bx
    return {"fr": fr, "fc": fc, "tr": tr, "tc": tc}


def best_move_aichess(fen: str, model_path: str | None, use_mcts: bool = True, n_playout: int = 1200, c_puct: int = 5) -> Dict[str, int]:
    # Build board
    board, side = _fen_to_aichess_board(fen)
    # Load policy-value net
    from aichess.pytorch_net import PolicyValueNet
    pv = PolicyValueNet(model_file=model_path) if model_path else PolicyValueNet()
    if use_mcts:
        from aichess.mcts import MCTSPlayer
        player = MCTSPlayer(pv.policy_value_fn, c_puct=c_puct, n_playout=n_playout, is_selfplay=0)
        # aichess board class used in MCTSPlayer expects their Board object
        move_id = player.get_action(board, temp=1e-3, return_prob=0)
    else:
        # Stateless greedy pick from policy
        legal = board.availables
        act_probs, _ = pv.policy_value(board.current_state().reshape(-1, 9, 10, 9))
        import numpy as np
        probs = np.zeros((2086,), dtype=np.float32)
        probs[legal] = act_probs.flatten()[legal]
        if probs.sum() <= 0:
            move_id = legal[0]
        else:
            move_id = int(probs.argmax())
    return _aichess_action_to_move_dict(move_id)


