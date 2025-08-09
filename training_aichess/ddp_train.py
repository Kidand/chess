#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple
import threading
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from tqdm import tqdm

from backend.xiangqi import parse_fen, to_fen, other, generate_legal_moves
from backend.encoding import board_to_planes
from training.az_mcts import AZMCTS, MCTSConfig
from training.az_model import XQAZNet
from training.az_aug import flip_planes_lr, flip_policy_lr
from training.eval_pool import BatchedEvaluator


START_FEN = "rheakaehr/9/1c5c1/s1s1s1s1s/9/9/S1S1S1S1S/1C5C1/9/RHEAKAEHR r"


class SimpleBuffer(Dataset):
    def __init__(self, capacity: int = 500_000):
        self.capacity = capacity
        self.states: List[np.ndarray] = []
        self.policies: List[np.ndarray] = []
        self.values: List[float] = []

    def push_game(self, samples: List[Tuple[np.ndarray, np.ndarray, float]]):
        for s, p, v in samples:
            self.states.append(s); self.policies.append(p); self.values.append(v)
        if len(self.states) > self.capacity:
            overflow = len(self.states) - self.capacity
            del self.states[:overflow]; del self.policies[:overflow]; del self.values[:overflow]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.states[idx]).float()
        p = torch.from_numpy(self.policies[idx]).float()
        v = torch.tensor([self.values[idx]], dtype=torch.float32)
        return x, p, v


def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def mcts_selfplay_game(net: torch.nn.Module, device: torch.device, cfg: MCTSConfig, evaluator: BatchedEvaluator,
                       temperature_moves: int = 12, no_capture_draw_plies: int = 60) -> Tuple[List[Tuple[np.ndarray,np.ndarray,float]], dict]:
    b, side = parse_fen(START_FEN)
    fen = START_FEN
    data = []
    no_cap = 0
    caps = 0
    mcts = AZMCTS(net, device, cfg)
    for ply in range(512):
        planes = board_to_planes(b, side)
        temp = 1.0 if ply < temperature_moves else 0.0
        # evaluator is attached inside AZMCTS in prior design; here AZMCTS uses net directly
        visits, action, _ = mcts.run(fen, temperature=temp)
        pi = visits / (visits.sum() + 1e-12)
        data.append((planes, pi, 0.0))
        frfc = action // 90; trtc = action % 90
        fr, fc = divmod(frfc, 9); tr, tc = divmod(trtc, 9)
        legals = generate_legal_moves(b, side)
        mv = None
        for m in legals:
            if m.from_row==fr and m.from_col==fc and m.to_row==tr and m.to_col==tc:
                mv = m; break
        if mv is None:
            if not legals:
                z = -1.0
                return [(s,p,z) for (s,p,_) in data], {"result": z, "plies": ply+1, "caps": caps}
            mv = legals[0]
        cap = (b[tr][tc] != '.')
        if cap: caps += 1
        b[tr][tc] = b[fr][fc]; b[fr][fc] = '.'
        side = other(side)
        fen = to_fen(b, side)
        if cap: no_cap = 0
        else:
            no_cap += 1
            if no_cap >= no_capture_draw_plies:
                return [(s,p,0.0) for (s,p,_) in data], {"result": 0.0, "plies": ply+1, "caps": caps}
        if not generate_legal_moves(b, side):
            z = 1.0
            return [(s,p,z) for (s,p,_) in data], {"result": z, "plies": ply+1, "caps": caps}
    return [(s,p,0.0) for (s,p,_) in data], {"result": 0.0, "plies": 512, "caps": caps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='runs/xq_aichess')
    parser.add_argument('--segments', type=int, default=16)
    parser.add_argument('--selfplay_per_seg', type=int, default=1024)
    parser.add_argument('--channels', type=int, default=256)
    parser.add_argument('--blocks', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_simulations', type=int, default=800)
    parser.add_argument('--envs_per_rank', type=int, default=8)
    args = parser.parse_args()

    setup_ddp()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device('cuda', rank % torch.cuda.device_count())

    out_dir = Path(args.out)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    net = XQAZNet(channels=args.channels, blocks=args.blocks).to(device)
    net = DDP(net, device_ids=[device.index]) if world > 1 else net
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    def loss_and_kl(p_logits_old, p_logits_new):
        # KL(new||old)
        p_old = torch.softmax(p_logits_old, dim=-1)
        p_new = torch.softmax(p_logits_new, dim=-1)
        kl = torch.mean(torch.sum(p_new * (torch.log(p_new+1e-10) - torch.log(p_old+1e-10)), dim=-1))
        return kl

    buffer = SimpleBuffer()
    mcts_cfg = MCTSConfig(num_simulations=args.num_simulations)

    for seg in range(1, args.segments+1):
        # self-play collection (multi-thread per rank)
        produced = 0
        lock = threading.Lock()
        errors: List[Exception] = []
        pbar = tqdm(total=args.selfplay_per_seg, desc=f'Seg {seg} Self-Play', dynamic_ncols=True) if rank == 0 else None
        stats = {"wins": 0, "draws": 0, "losses": 0}
        caps_sum = 0.0; plies_sum = 0.0
        evaluator = BatchedEvaluator(net.module if isinstance(net, DDP) else net, device, max_batch=512)

        def worker(wid: int):
            nonlocal produced
            try:
                while True:
                    with lock:
                        if produced >= args.selfplay_per_seg:
                            break
                        produced += 1
                    samples, info = None, None
                    game, info = mcts_selfplay_game(net.module if isinstance(net, DDP) else net, device, mcts_cfg, evaluator)
                    # augment
                    aug = []
                    for s, p, z in game:
                        aug.append((s, p, z))
                        aug.append((flip_planes_lr(s), flip_policy_lr(p), z))
                    with lock:
                        buffer.push_game(aug)
                        plies_sum += info.get('plies', 0.0)
                        caps_sum += info.get('caps', 0.0)
                        z = info.get('result', 0.0)
                        if z > 0: stats['wins'] += 1
                        elif z < 0: stats['losses'] += 1
                        else: stats['draws'] += 1
                        if pbar:
                            avg_p = plies_sum / max(1, produced)
                            avg_c = caps_sum / max(1, produced)
                            pbar.set_postfix_str(f"W/D/L={stats['wins']}/{stats['draws']}/{stats['losses']} avg_plies={avg_p:.1f} avg_caps={avg_c:.2f} buf={len(buffer)}")
                            pbar.update(1)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.envs_per_rank)]
        for t in threads: t.start()
        for t in threads: t.join()
        evaluator.close()
        if pbar: pbar.close()
        if errors: raise errors[0]

        # training on buffer
        dataset = buffer
        sampler = DistributedSampler(dataset) if world > 1 else None
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None), num_workers=4, pin_memory=True)
        net.train()
        if sampler is not None:
            sampler.set_epoch(seg)
        steps = 0; t0 = time.time()
        it = tqdm(loader, desc=f'Seg {seg} Train', dynamic_ncols=True) if rank == 0 else loader
        for x, p, v in it:
            x = x.to(device); p = p.to(device); v = v.to(device)
            with torch.no_grad():
                p_logits_old, _ = net(x)
            p_logits, v_pred = net(x)
            ce = torch.nn.functional.cross_entropy(p_logits, p.argmax(dim=-1))
            mse = torch.nn.functional.mse_loss(v_pred, v.squeeze(-1))
            loss = ce + mse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()
            # KL (monitor)
            with torch.no_grad():
                kl = loss_and_kl(p_logits_old, p_logits).item()
            steps += 1
        if rank == 0:
            dt = time.time()-t0
            tqdm.write(f"[Seg {seg}] train steps={steps} time={dt:.1f}s it/s={(steps/dt if dt>0 else 0):.1f} buf={len(buffer)}")
            ckpt = {'state_dict': (net.module.state_dict() if isinstance(net, DDP) else net.state_dict())}
            torch.save(ckpt, out_dir / f'aichess_seg_{seg}.pt')


if __name__ == '__main__':
    main()


