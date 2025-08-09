#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, pickle, time, random
from collections import deque
import numpy as np
from tqdm import tqdm

from training_aichess.config import CONFIG
from training_aichess.pytorch_net import PolicyValueNet
from training_aichess.mcts import MCTSPlayer
from backend.xiangqi import parse_fen, to_fen, other, generate_legal_moves
from backend.encoding import board_to_planes

START_FEN = "rheakaehr/9/1c5c1/s1s1s1s1s/9/9/S1S1S1S1S/1C5C1/9/RHEAKAEHR r"


class CollectPipeline:
    def __init__(self):
        self.data_buffer = deque(maxlen=CONFIG['buffer_size'])
        self.iters = 0
        self.net = PolicyValueNet(model_file=CONFIG['pytorch_model_path'])
        self.player = MCTSPlayer(self.net.policy_value_fn, c_puct=CONFIG['c_puct'], n_playout=CONFIG['play_out'], is_selfplay=1)

    def current_state(self, b, side):
        return board_to_planes(b, side)

    def self_play_one(self, temp=1.0):
        b, side = parse_fen(START_FEN)
        states = []
        mcts_probs = []
        winners = []
        result = 0
        for t in range(512):
            s = self.current_state(b, side)
            move, visits = self.player.get_action(b, side, temp=temp)
            pi = visits / (visits.sum()+1e-12)
            states.append(s); mcts_probs.append(pi)
            frfc = move // 90; trtc = move % 90
            fr, fc = divmod(frfc, 9); tr, tc = divmod(trtc, 9)
            legals = generate_legal_moves(b, side)
            chosen = None
            for m in legals:
                if m.from_row==fr and m.from_col==fc and m.to_row==tr and m.to_col==tc:
                    chosen = m; break
            if chosen is None:
                if not legals:
                    z = -1.0; winners = [z]*len(states); result = -1; break
                chosen = legals[0]
            cap = (b[tr][tc] != '.')
            b[tr][tc] = b[fr][fc]; b[fr][fc]='.'
            side = other(side)
            if not generate_legal_moves(b, side):
                z = 1.0; winners = [z]*len(states); result = 1; break
        if not winners:
            winners = [0.0]*len(states); result = 0
        return list(zip(states, mcts_probs, winners)), result

    def run(self):
        pbar = tqdm(total=0, desc='Collect', dynamic_ncols=True)
        while True:
            game, result = self.self_play_one(temp=1.0)
            self.data_buffer.extend(game)
            self.iters += 1
            data = {'data_buffer': self.data_buffer, 'iters': self.iters}
            with open(CONFIG['train_data_buffer_path'], 'wb') as f:
                pickle.dump(data, f)
            pbar.total = self.iters
            pbar.n = self.iters
            outcome = 'W' if result>0 else ('L' if result<0 else 'D')
            pbar.set_postfix_str(f"{outcome} game_len={len(game)} buffer={len(self.data_buffer)}")
            pbar.refresh()


if __name__=='__main__':
    CollectPipeline().run()


