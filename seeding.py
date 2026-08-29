#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Seeding utilities for reproducible (but not bit-exact) training.

NOTE: We deliberately do NOT call torch.use_deterministic_algorithms(True)
per user's request (it forces deterministic kernels for every op, which can
error out on ops without a deterministic implementation, and slows training).
What we do instead gets you "practically deterministic" results:
    - identical data order every run
    - identical model init every run
    - identical dropout masks every run
    - cudnn deterministic algo selection (not full determinism, but stable)
Exact bit-for-bit reproducibility across GPUs/driver versions is still not
guaranteed without use_deterministic_algorithms(True), but run-to-run
variance on the same machine/driver will be effectively eliminated.
"""
import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ask cuDNN to pick deterministic algorithms where available, without
    # the hard-fail behavior of use_deterministic_algorithms(True).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Pass as worker_init_fn to DataLoader so each dataloader worker gets a
    deterministic (but distinct) seed derived from torch's initial seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
