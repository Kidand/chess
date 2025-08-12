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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
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
        # Safer NCCL defaults
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
        os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")

        use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
        backend = "nccl" if use_cuda else "gloo"

        # Set device BEFORE initializing process group to avoid device-id warnings and mismatches
        if use_cuda:
            local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
            local_rank = local_rank % torch.cuda.device_count()
            torch.cuda.set_device(local_rank)

        dist.init_process_group(backend=backend, timeout=timedelta(minutes=60))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='runs/xq')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--segments', type=int, default=16)
    parser.add_argument('--selfplay_per_seg', type=int, default=1024)
    parser.add_argument('--channels', type=int, default=256)
    parser.add_argument('--blocks', type=int, default=12)
    parser.add_argument('--policy_head', type=str, default='flat', choices=['flat','structured'])
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
        local_rank = int(os.environ.get("LOCAL_RANK", rank)) % torch.cuda.device_count()
        # torch.cuda.set_device was already called in setup_ddp, but call again for safety
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    out_dir = Path(args.out).resolve()
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    net = XQAZNet(channels=args.channels, blocks=args.blocks, policy_head=args.policy_head).to(device)
    resume_seg_offset = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        sd = ckpt.get('state_dict', ckpt)
        net.load_state_dict(sd, strict=False)
        try:
            resume_seg_offset = int(ckpt.get('seg', 0) or 0)
        except Exception:
            resume_seg_offset = 0
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
        # Soft-label cross-entropy for policy: -sum(pi * log softmax(logits))
        target_p = target_p / target_p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        logp = F.log_softmax(p_logits, dim=-1)
        policy_loss = -(target_p * logp).sum(dim=-1).mean()
        value_loss = F.mse_loss(v_pred, target_v.squeeze(-1))
        return policy_loss + value_loss

    # Replay buffer
    rb = ReplayBuffer(capacity=500_000)

    # helper: build per-segment MCTSConfig with tiered capture shaping and annealing
    def build_mcts_cfg(seg_idx: int) -> MCTSConfig:
        # Use global segment index across resumes so that Stage B continues Stage A's annealing
        global_seg = int(resume_seg_offset) + int(seg_idx)
        # Segment-based annealing: 1-4 strong, 5-8 half, >=9 off
        if global_seg <= 4:
            cap_mode = 'tiered'; cap_tier_scale = 1.0; check_boost = 1.6
        elif global_seg <= 8:
            cap_mode = 'tiered'; cap_tier_scale = 0.5; check_boost = 1.3
        else:
            cap_mode = 'uniform'; cap_tier_scale = 0.0; check_boost = 1.0
        cap_tier = {'R': 2.0, 'C': 1.6, 'H': 1.5, 'S': 1.2, 'A': 1.1, 'E': 1.1}
        return MCTSConfig(
            num_simulations=args.num_simulations,
            cpuct=1.8 if global_seg <= 4 else (1.6 if global_seg <= 8 else 1.5),
            dirichlet_alpha=0.3,
            dirichlet_frac=0.33 if global_seg <= 4 else (0.25 if global_seg <= 8 else 0.15),
            fpu_value=-0.1 if global_seg <= 4 else (-0.15 if global_seg <= 8 else -0.2),
            cap_boost=2.0,
            check_boost=check_boost,
            cap_mode=cap_mode,
            cap_tier=cap_tier,
            cap_tier_scale=cap_tier_scale,
        )
    mcts_cfg = build_mcts_cfg(1)

    # segments loop
    for seg in range(1, args.segments + 1):
        # self-play with threads per rank
        total_games = args.selfplay_per_seg
        import threading
        lock = threading.Lock()
        produced = 0  # scheduled games on this rank
        completed = 0  # finished games on this rank
        errors: List[Exception] = []
        # Global progress bar on rank 0 shows aggregated progress across all ranks
        global_total = total_games * world
        pbar = tqdm(total=global_total, desc=f'Self-Play Seg {seg}', dynamic_ncols=True) if rank == 0 else None
        stats = {"red_wins": 0, "black_wins": 0, "draws": 0}
        caps_sum = 0.0; plies_sum = 0.0

        def worker(wid: int):
            nonlocal produced, completed, plies_sum, caps_sum, stats
            try:
                while True:
                    with lock:
                        if produced >= total_games:
                            break
                        produced += 1
                    # refresh MCTS config by segment to apply annealing
                    mcts_cfg_local = build_mcts_cfg(seg)
                    samples, z, info = play_one_game(net.module if isinstance(net, DDP) else net, device, mcts_cfg_local,
                                               args.temperature_moves, args.no_capture_draw_plies)
                    # fill final z to each sample (no bootstrap for simplicity in v1)
                    game = []
                    # Train value from side-to-move perspective at each ply.
                    # Starting side is red; flip sign for odd plies (black to move).
                    for ply_idx, s in enumerate(samples):
                        v_tgt = z if (ply_idx % 2 == 0) else -z
                        game.append(Sample(planes=s.planes, policy=s.policy, value=v_tgt))
                        # left-right flip augmentation
                        game.append(Sample(planes=flip_planes_lr(s.planes), policy=flip_policy_lr(s.policy), value=v_tgt))
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
                        # assign per-rank sequential index for finished games
                        current_idx = completed + 1
                        completed += 1
                        # dump win/loss to per-rank jsonl under out_dir/datasets/seg_{seg}
                        try:
                            if z != 0.0:
                                rec = {
                                    'timestamp': datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                                    'seg': int(seg),
                                    'rank': int(rank),
                                    'idx': int(current_idx),
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
                                base_dir = (out_dir / 'datasets' / f'seg_{seg}')
                                base_dir.mkdir(parents=True, exist_ok=True)
                                shard = f"win_rank{rank}.jsonl" if z > 0 else f"loss_rank{rank}.jsonl"
                                out_path = base_dir / shard
                                with open(out_path, 'a', encoding='utf-8') as f:
                                    f.write(json.dumps(rec, ensure_ascii=False)+"\n")
                        except Exception:
                            pass
            except Exception as e:
                with lock:
                    errors.append(e)

        # Aggregated progress monitor across ranks (all ranks participate; rank 0 displays)
        def monitor_fn():
            import time as _time
            while True:
                # snapshot local counters
                with lock:
                    lp = float(completed)
                    lp_plies = float(plies_sum)
                    lp_caps = float(caps_sum)
                    lr = float(stats['red_wins'])
                    lb = float(stats['black_wins'])
                    ld = float(stats['draws'])
                vec = torch.tensor([lp, lp_plies, lp_caps, lr, lb, ld], dtype=torch.float64, device=(device if device.type=='cuda' else torch.device('cpu')))
                if world > 1:
                    dist.all_reduce(vec, op=dist.ReduceOp.SUM)
                gp, gplies, gcaps, gr, gb, gd = vec.tolist()
                # update bar and postfix (rank 0 only)
                if pbar:
                    target_total = global_total
                    pbar.total = target_total
                    # Increment to global completed
                    delta = int(gp) - pbar.n
                    if delta > 0:
                        pbar.update(delta)
                    avg_p = (gplies / gp) if gp > 0 else 0.0
                    avg_c = (gcaps / gp) if gp > 0 else 0.0
                    pbar.set_postfix_str(f"R/B/D={int(gr)}/{int(gb)}/{int(gd)} avg_plies={avg_p:.1f} avg_caps={avg_c:.2f}")
                # exit when all ranks have finished their quotas
                if int(gp) >= global_total:
                    break
                _time.sleep(0.5)

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.envs_per_rank)]
        for t in threads: t.start()
        # Main-thread progress aggregation loop (collectives are only called here to keep order identical across ranks)
        import time as _time
        while True:
            # snapshot local counters
            with lock:
                lp = float(completed)
                lp_plies = float(plies_sum)
                lp_caps = float(caps_sum)
                lr = float(stats['red_wins'])
                lb = float(stats['black_wins'])
                ld = float(stats['draws'])
            vec = torch.tensor([lp, lp_plies, lp_caps, lr, lb, ld], dtype=torch.float64, device=(device if device.type=='cuda' else torch.device('cpu')))
            if world > 1:
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)
            gp, gplies, gcaps, gr, gb, gd = vec.tolist()
            if pbar:
                pbar.total = global_total
                delta = int(gp) - pbar.n
                if delta > 0:
                    pbar.update(delta)
                avg_p = (gplies / gp) if gp > 0 else 0.0
                avg_c = (gcaps / gp) if gp > 0 else 0.0
                pbar.set_postfix_str(f"R/B/D={int(gr)}/{int(gb)}/{int(gd)} avg_plies={avg_p:.1f} avg_caps={avg_c:.2f}")
            # exit condition: all threads finished and global progress reached total
            if not any(t.is_alive() for t in threads) and int(gp) >= global_total:
                break
            _time.sleep(0.5)
        for t in threads: t.join()
        if pbar: pbar.close()
        if errors: raise errors[0]
        if world > 1:
            dist.barrier()
        # Aggregate per-rank JSONL shards into a single file per result on rank 0
        try:
            if rank == 0:
                seg_dir = out_dir / 'datasets' / f'seg_{seg}'
                if seg_dir.exists():
                    for kind in ('win', 'loss'):
                        merged = seg_dir / f'{kind}.jsonl'
                        # truncate old merged to avoid duplication if re-run
                        with open(merged, 'w', encoding='utf-8') as fout:
                            for shard in sorted(seg_dir.glob(f'{kind}_rank*.jsonl')):
                                with open(shard, 'r', encoding='utf-8') as fin:
                                    for line in fin:
                                        fout.write(line)
        except Exception:
            pass

        # train epochs for this segment
        # Ensure identical dataset length across ranks to avoid collective mismatches
        class _RBSubset(Dataset):
            def __init__(self, base: ReplayBuffer, length: int):
                self.base = base
                self.length = max(0, min(length, len(base)))
            def __len__(self) -> int:
                return self.length
            def __getitem__(self, idx: int):
                return self.base[idx]

        if world > 1:
            # Gather per-rank lengths and take the global minimum
            local_len = len(rb)
            all_lens = [0 for _ in range(world)]
            dist.all_gather_object(all_lens, local_len)
            min_len = min(int(x) for x in all_lens)
            # Truncate to a multiple of (world * batch_size) to ensure identical number of full batches per rank
            global_batch = world * args.batch_size
            max_full = (min_len // global_batch) * global_batch
            if max_full == 0:
                # No full batch available; skip this segment's training
                if rank == 0:
                    tqdm.write(f"[Train] seg={seg} skipped (insufficient samples across ranks: min_len={min_len})")
                dist.barrier()
                continue
            dataset = _RBSubset(rb, max_full)
            sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        else:
            dataset = rb
            sampler = None
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=4,
            pin_memory=(device.type == 'cuda'),
            drop_last=(world > 1),
        )
        for ep in range(args.epochs):
            net.train()
            if sampler is not None:
                sampler.set_epoch(ep + seg)
                dist.barrier()
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


