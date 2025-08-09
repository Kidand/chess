#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batched evaluator to aggregate many inference requests into large GPU batches.

English comments per user preference.
"""

from __future__ import annotations

import threading
import time
from queue import Queue, Empty
from typing import Any, Tuple

import numpy as np
import torch


class _Future:
    def __init__(self):
        self._evt = threading.Event()
        self._result: Tuple[np.ndarray, float] | None = None

    def set_result(self, value: Tuple[np.ndarray, float]):
        self._result = value
        self._evt.set()

    def result(self) -> Tuple[np.ndarray, float]:
        self._evt.wait()
        assert self._result is not None
        return self._result


class BatchedEvaluator:
    def __init__(self, net: torch.nn.Module, device: torch.device, max_batch: int = 256, max_wait_ms: int = 5):
        self.net = net
        self.device = device
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self.q: Queue[Tuple[np.ndarray, _Future]] = Queue()
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._loop, daemon=True)
        # stats
        self._lock = threading.Lock()
        self._batches = 0
        self._total_items = 0
        self._thr.start()

    def submit(self, planes: np.ndarray) -> _Future:
        fut = _Future()
        self.q.put((planes, fut))
        return fut

    def close(self):
        self._stop.set()
        self._thr.join(timeout=1.0)

    def _loop(self):
        while not self._stop.is_set():
            items: list[Tuple[np.ndarray, _Future]] = []
            # block for first item
            try:
                item = self.q.get(timeout=0.01)
                items.append(item)
            except Empty:
                continue
            # then drain up to max_batch or until wait time
            t0 = time.time()
            while len(items) < self.max_batch:
                remain = self.max_wait_ms/1000 - (time.time()-t0)
                if remain <= 0:
                    break
                try:
                    items.append(self.q.get(timeout=max(0.0, remain)))
                except Empty:
                    break
            # run batch
            planes_list = [p for p,_ in items]
            x = torch.from_numpy(np.stack(planes_list, axis=0)).to(self.device)
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
            with torch.no_grad():
                with ac:
                    p_logits, v = self.net(x)
                    p = torch.softmax(p_logits, dim=-1).float().cpu().numpy()
                    vv = v.float().cpu().numpy()
            with self._lock:
                self._batches += 1
                self._total_items += len(items)
            # return results
            for i, (_, fut) in enumerate(items):
                fut.set_result((p[i], float(vv[i])))

    def get_and_reset_stats(self) -> tuple[int, int, float]:
        """Return (num_batches, total_items, avg_batch) and reset counters."""
        with self._lock:
            b = self._batches
            t = self._total_items
            self._batches = 0
            self._total_items = 0
        avg = (t / b) if b > 0 else 0.0
        return b, t, avg


