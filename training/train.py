#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AlphaZero training script for Xiangqi with multi-GPU (H100) support.

Usage (single node, 8 GPUs):
  torchrun --standalone --nproc_per_node=8 training/train.py --out runs/xq --selfplay_steps 1000 --epochs 1

English comments per user preference.
"""

from __future__ import annotations

# Ensure project root is importable when running as a script (torchrun with script path)
import os, sys
_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import os
import math
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import time

try:
    from .model import XQAlphaZeroNet
    from .dataset import ReplayDataset, Sample
    from .selfplay import play_one_game
except Exception:
    # Allow running as script: python -m training.train
    from training.model import XQAlphaZeroNet  # type: ignore
    from training.dataset import ReplayDataset, Sample  # type: ignore
    from training.selfplay import play_one_game  # type: ignore


def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def gather_all_samples(local_samples: List[Sample]) -> List[Sample]:
    # naive CPU-side gather via file not implemented here; using DDP, each rank will contribute equally
    return local_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--channels', type=int, default=256)
    parser.add_argument('--blocks', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--selfplay_steps', type=int, default=1000)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--mcts_sims', type=int, default=800, help='MCTS simulations per move during self-play')
    parser.add_argument('--mcts_batch', type=int, default=256, help='Batched leaf evaluations per forward')
    parser.add_argument('--max_moves', type=int, default=256, help='Max moves per self-play game')
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--segment_k', type=int, default=2000, help='Train/ckpt every K self-play games per rank')
    parser.add_argument('--envs_per_rank', type=int, default=4, help='Parallel self-play environments per rank')
    parser.add_argument('--log_steps', action='store_true', help='Log every self-play step')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume (seg_*.pt or model_epoch_*.pt)')
    parser.add_argument('--resume_latest', action='store_true', help='Resume from latest ckpt found in --out directory')
    args = parser.parse_args()

    setup_ddp()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1

    out_dir = Path(args.out)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device('cuda', rank % torch.cuda.device_count())

    net = XQAlphaZeroNet(channels=args.channels, num_blocks=args.blocks).to(device)
    net = DDP(net, device_ids=[device.index]) if world > 1 else net

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    # loss: policy CE + value MSE (+ optional bootstrap + reward shaping)
    def loss_fn(p_logits, v_pred, target_p, target_v):
        ce = torch.nn.functional.cross_entropy(p_logits, target_p.argmax(dim=-1))
        mse = torch.nn.functional.mse_loss(v_pred, target_v.squeeze(-1))
        return ce + mse

    # knobs for signal shaping
    reward_w = 0.01  # small weight on per-step reward
    bootstrap_w = 0.1  # weight on value bootstrap

    # optional resume
    def _load_ckpt(path: str):
        ckpt = torch.load(path, map_location=device)
        sd = ckpt.get('state_dict', ckpt)
        (net.module if isinstance(net, DDP) else net).load_state_dict(sd, strict=False)
        if 'optimizer' in ckpt:
            try:
                opt.load_state_dict(ckpt['optimizer'])
            except Exception:
                pass
        if rank == 0:
            tqdm.write(f"[Rank0] Resumed from {path}")

    if args.resume_latest and rank == 0:
        # find newest ckpt in out_dir (segments or epoch)
        candidates = []
        for p in out_dir.glob('model_epoch_*.pt'):
            candidates.append(p)
        for p in (out_dir / 'segments').glob('seg_*.pt') if (out_dir / 'segments').exists() else []:
            candidates.append(p)
        if candidates:
            latest = max(candidates, key=lambda p: p.stat().st_mtime)
            args.resume = str(latest)
    # broadcast resume path
    if dist.is_initialized():
        import torch.distributed as _dist
        t = torch.tensor([len(args.resume)], dtype=torch.int32, device=device)
        _dist.broadcast(t, src=0)
        n = int(t.item())
        if rank != 0:
            args.resume = ''
        if n > 0:
            if rank == 0:
                b = torch.tensor(list(bytearray(args.resume, 'utf-8')), dtype=torch.uint8, device=device)
            else:
                b = torch.empty(n, dtype=torch.uint8, device=device)
            _dist.broadcast(b, src=0)
            if rank != 0:
                args.resume = bytes(b.tolist()).decode('utf-8')
    if args.resume:
        _load_ckpt(args.resume)

    # segmented self-play/train alternating
    local_samples: List[Sample] = []
    total_sp = args.selfplay_steps
    seg = max(1, args.segment_k)
    rounds = (total_sp + seg - 1) // seg
    sp_done = 0
    for round_idx in range(rounds):
        to_play = min(seg, total_sp - sp_done)
        # parallel self-play workers per rank sharing one evaluator
        with torch.no_grad():
            net.eval()
            from training.eval_pool import BatchedEvaluator
            evaluator = BatchedEvaluator(net.module if isinstance(net, DDP) else net, device, max_batch=args.mcts_batch)

            import threading
            lock = threading.Lock()
            produced = 0
            errors = []
            pbar = tqdm(total=to_play, desc=f'Self-Play R{round_idx+1}/{rounds}', dynamic_ncols=True) if rank == 0 else None
            t_start = time.time()

            def log_fn(msg: str):
                if rank == 0:
                    tqdm.write(msg)

            def worker(worker_id: int):
                nonlocal produced
                try:
                    while True:
                        with lock:
                            if produced >= to_play:
                                break
                            produced += 1
                            idx = produced
                        sp_data, z_final = play_one_game(
                            net.module if isinstance(net, DDP) else net,
                            device,
                            temperature=args.temp,
                            temp_init=1.0,
                            temp_final=0.0,
                            temp_decay_moves=16,
                            max_moves=args.max_moves,
                            mcts_sims=args.mcts_sims,
                            mcts_batch=args.mcts_batch,
                            evaluator=evaluator,
                            log_steps=args.log_steps,
                            log_result=True,
                            log_fn=log_fn,
                            game_tag=f"R{round_idx+1}-T{worker_id}-G{idx}",
                            progress=(rank == 0 and (idx % 10 == 0)),
                        )
                        with lock:
                            # compose targets: mix final result z with bootstrap and reward shaping
                            # sp_data: (planes, pi, v0, reward)
                            val_target = z_final
                            # construct per-step value targets
                            for si, (planes, pi, v0, rew) in enumerate(sp_data):
                                # bootstrapped value blends with final outcome
                                vt = (1.0 - bootstrap_w) * val_target + bootstrap_w * v0
                                # add small step reward (signed by side-to-move alternation)
                                signed_rew = ((1 if (si % 2 == 0) else -1) * rew)
                                vt = vt + reward_w * signed_rew
                                local_samples.append(Sample(planes=planes, policy=pi, value=vt))
                            if pbar: pbar.update(1)
                except Exception as e:
                    with lock:
                        errors.append(e)

            threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.envs_per_rank)]
            for t in threads: t.start()
            for t in threads: t.join()
            # stats
            b, t_items, avg = evaluator.get_and_reset_stats()
            elapsed = time.time() - t_start
            if rank == 0:
                tqdm.write(f"[Rank0] SP seg {round_idx+1}: {to_play} games, eval_batches={b}, eval_items={t_items}, avg_batch={avg:.1f}, time={elapsed:.1f}s, ips={(t_items/elapsed):.1f}")
            evaluator.close()
            if pbar: pbar.close()
            if errors:
                raise errors[0]
        sp_done += to_play

        # one short training epoch after each segment
        net.train()
        if world > 1 and isinstance(net, DDP):
            dist.barrier()
        dataset = ReplayDataset(local_samples)
        sampler = DistributedSampler(dataset) if world > 1 else None
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None), num_workers=4, pin_memory=True)

        iter_loader = loader
        if rank == 0:
            iter_loader = tqdm(loader, desc=f'Train Seg {round_idx+1}', dynamic_ncols=True)
        t0 = time.time(); steps=0
        for x, p, v in iter_loader:
            x = x.to(device)
            p = p.to(device)
            v = v.to(device)
            p_logits, v_pred = net(x)
            loss = loss_fn(p_logits, v_pred, p, v)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()
            steps += 1
        if rank == 0:
            dt = time.time()-t0
            tqdm.write(f"[Rank0] Train seg {round_idx+1}: steps={steps}, time={dt:.1f}s, it/s={(steps/dt if dt>0 else 0):.1f}")

        if rank == 0:
            ckpt = {
                'round': round_idx + 1,
                'state_dict': (net.module.state_dict() if isinstance(net, DDP) else net.state_dict()),
                'optimizer': opt.state_dict(),
            }
            (out_dir / 'segments').mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, out_dir / 'segments' / f"seg_{round_idx+1}.pt")
    # (optional) gather across ranks; here we keep per-rank shards

    # training dataloader
    dataset = ReplayDataset(local_samples)
    sampler = DistributedSampler(dataset) if world > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None), num_workers=4, pin_memory=True)

    for epoch in range(1, args.epochs + 1):
        net.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        iter_loader = loader
        if rank == 0:
            iter_loader = tqdm(loader, desc=f'Train Epoch {epoch}', dynamic_ncols=True)
        for x, p, v in iter_loader:
            x = x.to(device)
            p = p.to(device)
            v = v.to(device)
            p_logits, v_pred = net(x)
            loss = loss_fn(p_logits, v_pred, p, v)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()
        if rank == 0 and (epoch % args.save_every == 0):
            ckpt = {
                'epoch': epoch,
                'state_dict': (net.module.state_dict() if isinstance(net, DDP) else net.state_dict()),
                'optimizer': opt.state_dict(),
            }
            torch.save(ckpt, out_dir / f"model_epoch_{epoch}.pt")

    cleanup_ddp()


if __name__ == '__main__':
    main()


