#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AlphaZero training entry (clean redesign).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from .az_config import AZConfig
from .az_model import XQAZNet
from .az_mcts import MCTSConfig
from .az_replay import ReplayBuffer, Sample
from .az_selfplay import play_one_game


def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='runs/xq')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--segments', type=int, default=16)
    parser.add_argument('--selfplay_per_seg', type=int, default=1024)
    parser.add_argument('--channels', type=int, default=256)
    parser.add_argument('--blocks', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_simulations', type=int, default=800)
    parser.add_argument('--mcts_batch', type=int, default=512)
    parser.add_argument('--envs_per_rank', type=int, default=8)
    parser.add_argument('--temperature_moves', type=int, default=20)
    parser.add_argument('--no_capture_draw_plies', type=int, default=60)
    parser.add_argument('--resume', type=str, default='')
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
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        sd = ckpt.get('state_dict', ckpt)
        net.load_state_dict(sd, strict=False)
    net = DDP(net, device_ids=[device.index]) if world > 1 else net
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    def loss_fn(p_logits, v_pred, target_p, target_v):
        ce = torch.nn.functional.cross_entropy(p_logits, target_p.argmax(dim=-1))
        mse = torch.nn.functional.mse_loss(v_pred, target_v.squeeze(-1))
        return ce + mse

    # Replay buffer
    rb = ReplayBuffer(capacity=500_000)

    mcts_cfg = MCTSConfig(num_simulations=args.num_simulations)

    # segments loop
    for seg in range(1, args.segments + 1):
        # self-play with threads per rank
        total_games = args.selfplay_per_seg
        import threading
        lock = threading.Lock()
        produced = 0
        errors: List[Exception] = []
        pbar = tqdm(total=total_games, desc=f'Self-Play Seg {seg}', dynamic_ncols=True) if rank == 0 else None
        stats = {"wins": 0, "draws": 0, "losses": 0}
        caps_sum = 0.0; plies_sum = 0.0

        def worker(wid: int):
            nonlocal produced
            try:
                while True:
                    with lock:
                        if produced >= total_games:
                            break
                        produced += 1
                    samples, z, info = play_one_game(net.module if isinstance(net, DDP) else net, device, mcts_cfg,
                                               args.temperature_moves, args.no_capture_draw_plies)
                    # fill final z to each sample (no bootstrap for simplicity in v1)
                    game = []
                    for s in samples:
                        game.append(Sample(planes=s.planes, policy=s.policy, value=z))
                    with lock:
                        rb.push_game(game)
                        plies_sum += info.get('plies', 0.0)
                        caps_sum += info.get('caps', 0.0)
                        if z > 0: stats['wins'] += 1
                        elif z < 0: stats['losses'] += 1
                        else: stats['draws'] += 1
                        if pbar:
                            avg_p = plies_sum / max(1, produced)
                            avg_c = caps_sum / max(1, produced)
                            pbar.set_postfix_str(f"W/D/L={stats['wins']}/{stats['draws']}/{stats['losses']} avg_plies={avg_p:.1f} avg_caps={avg_c:.2f}")
                            pbar.update(1)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.envs_per_rank)]
        for t in threads: t.start()
        for t in threads: t.join()
        if pbar: pbar.close()
        if errors: raise errors[0]

        # train epochs for this segment
        dataset = rb
        sampler = DistributedSampler(dataset) if world > 1 else None
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None), num_workers=4, pin_memory=True)
        for ep in range(args.epochs):
            net.train()
            if sampler is not None:
                sampler.set_epoch(ep + seg)
            it = tqdm(loader, desc=f'Train Seg {seg} Ep {ep+1}', dynamic_ncols=True) if rank == 0 else loader
            steps = 0; t0 = time.time()
            for x, p, v in it:
                x = x.to(device); p = p.to(device); v = v.to(device)
                p_logits, v_pred = net(x)
                loss = loss_fn(p_logits, v_pred, p, v)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                opt.step()
                steps += 1
            if rank == 0:
                dt = time.time() - t0
                tqdm.write(f"[Train] seg={seg} ep={ep+1} steps={steps} time={dt:.1f}s it/s={(steps/dt if dt>0 else 0):.1f}")

        if rank == 0:
            ckpt = {
                'state_dict': (net.module.state_dict() if isinstance(net, DDP) else net.state_dict()),
                'optimizer': opt.state_dict(),
                'seg': seg,
            }
            torch.save(ckpt, out_dir / f'az_seg_{seg}.pt')

    if rank == 0:
        ckpt = {
            'state_dict': (net.module.state_dict() if isinstance(net, DDP) else net.state_dict()),
            'optimizer': opt.state_dict(),
        }
        torch.save(ckpt, out_dir / 'az_final.pt')


if __name__ == '__main__':
    main()


