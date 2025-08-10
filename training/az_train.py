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
from datetime import datetime, timezone
import json
import numpy as np

from .az_config import AZConfig
from .az_model import XQAZNet
from .az_mcts import MCTSConfig
from .az_replay import ReplayBuffer, Sample
from .az_selfplay import play_one_game, START_FEN
from .az_aug import flip_planes_lr, flip_policy_lr


def setup_ddp():
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)


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

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = torch.device('cuda', rank % torch.cuda.device_count())
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    out_dir = Path(args.out)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    net = XQAZNet(channels=args.channels, blocks=args.blocks).to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        sd = ckpt.get('state_dict', ckpt)
        net.load_state_dict(sd, strict=False)
    if world > 1:
        if device.type == 'cuda':
            net = DDP(net, device_ids=[device.index])
        else:
            net = DDP(net)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    # cosine LR schedule with warmup
    steps_per_epoch_est = 1000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.segments * args.epochs * steps_per_epoch_est, eta_min=args.lr * 0.1)

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
        stats = {"red_wins": 0, "black_wins": 0, "draws": 0}
        caps_sum = 0.0; plies_sum = 0.0

        def worker(wid: int):
            nonlocal produced, plies_sum, caps_sum, stats
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
                        # left-right flip augmentation
                        game.append(Sample(planes=flip_planes_lr(s.planes), policy=flip_policy_lr(s.policy), value=z))
                    with lock:
                        rb.push_game(game)
                        plies_sum += info.get('plies', 0.0)
                        caps_sum += info.get('caps', 0.0)
                        winner = info.get('winner', '')
                        if winner == 'r':
                            stats['red_wins'] += 1
                        elif winner == 'b':
                            stats['black_wins'] += 1
                        else:
                            stats['draws'] += 1
                        # dump win/loss to jsonl incrementally
                        try:
                            import os
                            if z != 0.0:
                                rec = {
                                    'timestamp': datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                                    'seg': int(seg),
                                    'rank': int(rank),
                                    'idx': int(produced),
                                    'result': int(np.sign(z)),
                                    'winner': info.get('winner', ''),
                                    'reason': int(info.get('reason', 0.0)),
                                    'plies': int(info.get('plies', 0.0)),
                                    'caps': int(info.get('caps', 0.0)),
                                    'moves': info.get('moves', []),
                                    'start_fen': START_FEN,
                                    'mcts': {
                                        'num_simulations': int(mcts_cfg.num_simulations),
                                        'cpuct': float(mcts_cfg.cpuct),
                                        'dirichlet_alpha': float(mcts_cfg.dirichlet_alpha),
                                        'dirichlet_frac': float(mcts_cfg.dirichlet_frac),
                                        'cap_boost': float(mcts_cfg.cap_boost),
                                        'check_boost': float(mcts_cfg.check_boost),
                                        'fpu_value': float(mcts_cfg.fpu_value),
                                        'c_base': float(mcts_cfg.c_base),
                                        'c_init': float(mcts_cfg.c_init),
                                    },
                                    'temperature_moves': int(args.temperature_moves),
                                    'no_capture_draw_plies': int(args.no_capture_draw_plies),
                                    'envs_per_rank': int(args.envs_per_rank),
                                    'mcts_batch': int(args.mcts_batch),
                                }
                                os.makedirs('datasets', exist_ok=True)
                                out_path = 'datasets/wl_games_win.jsonl' if z > 0 else 'datasets/wl_games_loss.jsonl'
                                with open(out_path, 'a') as f:
                                    f.write(json.dumps(rec, ensure_ascii=False)+"\n")
                        except Exception:
                            pass
                        if pbar:
                            avg_p = plies_sum / max(1, produced)
                            avg_c = caps_sum / max(1, produced)
                            pbar.set_postfix_str(f"R/B/D={stats['red_wins']}/{stats['black_wins']}/{stats['draws']} avg_plies={avg_p:.1f} avg_caps={avg_c:.2f}")
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
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=4,
            pin_memory=(device.type == 'cuda'),
        )
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
                scheduler.step()
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


