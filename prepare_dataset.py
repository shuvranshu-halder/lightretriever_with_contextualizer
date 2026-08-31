#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Stage 1 of 2: build a single, disk-budget-constrained, self-contained
training dataset ONCE, then delete HF's raw download cache to avoid the
"parquet download + Arrow cache" double-storage problem.

Output: a datasets.DatasetDict saved to --output_dir via save_to_disk(),
containing "train" and "validation" splits. train.py loads this directly
via --prepared_dataset_dir and does NOT touch the network for data again.

Usage (budget-constrained, recommended when disk is limited):
    python prepare_dataset.py \
        --data_mixture_config /path/to/lightretriever/config/data/exp-m.json \
        --disk_budget_gb 50 \
        --output_dir ./data/prepared

Usage (plain subset + percentage, no official mixture weighting):
    python prepare_dataset.py \
        --subsets msmarco nq \
        --dataset_percentage 20 \
        --output_dir ./data/prepared
"""
import os
import glob
import shutil
import argparse

from datasets import DatasetDict
from huggingface_hub import constants as hf_constants

from config import DataConfig
from dataset import load_finetune_data, load_finetune_data_with_budget


def cleanup_raw_hf_cache(dataset_name: str, verbose: bool = True) -> None:
    """Delete HF's raw download + intermediate Arrow-conversion cache for
    this dataset repo. Safe to call AFTER save_to_disk() has produced a
    self-contained copy in --output_dir - nothing further depends on this
    cache. Reclaims the "double storage" overhead."""
    freed_bytes = 0
    slug = dataset_name.replace("/", "--")

    candidates = []
    hub_cache = getattr(hf_constants, "HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    candidates += glob.glob(os.path.join(hub_cache, f"datasets--{slug}*"))

    datasets_cache = os.path.expanduser(os.environ.get("HF_DATASETS_CACHE", "~/.cache/huggingface/datasets"))
    candidates += glob.glob(os.path.join(datasets_cache, f"*{slug}*"))
    candidates += glob.glob(os.path.join(datasets_cache, "parquet", f"*{slug}*"))
    candidates += glob.glob(os.path.join(datasets_cache, "downloads", "*"))  # generic parquet download cache

    for path in set(candidates):
        try:
            size = sum(f.stat().st_size for f in __import__("pathlib").Path(path).rglob("*") if f.is_file())
            shutil.rmtree(path, ignore_errors=True)
            freed_bytes += size
        except Exception as e:
            if verbose:
                print(f"  [cleanup] could not remove {path}: {e}")

    if verbose:
        print(f"[cleanup] reclaimed ~{freed_bytes / (1024**3):.2f}GB of raw HF cache")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, default=DataConfig.dataset_name)

    # Budget-constrained path (recommended)
    p.add_argument("--data_mixture_config", type=str, default=None)
    p.add_argument("--disk_budget_gb", type=float, default=None)

    # Plain subset + percentage path (fallback if no mixture config)
    p.add_argument("--subsets", type=str, nargs="+", default=DataConfig.subsets)
    p.add_argument("--dataset_percentage", type=float, default=DataConfig.dataset_percentage)

    p.add_argument("--val_fraction", type=float, default=DataConfig.val_fraction)
    p.add_argument("--seed", type=int, default=DataConfig.seed)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--no_cleanup", action="store_true", help="Skip deleting the raw HF download cache afterward.")
    p.add_argument("--force", action="store_true", help="Rebuild even if --output_dir already has a prepared dataset.")
    args = p.parse_args()

    if os.path.exists(os.path.join(args.output_dir, "dataset_dict.json")) and not args.force:
        print(f"[prepare] {args.output_dir} already contains a prepared dataset. "
              f"Skipping (pass --force to rebuild).")
        return

    if args.data_mixture_config and args.disk_budget_gb:
        print(f"[prepare] Budget-constrained build: {args.disk_budget_gb}GB via {args.data_mixture_config}")
        train_ds, val_ds = load_finetune_data_with_budget(
            args.dataset_name, args.data_mixture_config, args.disk_budget_gb, args.val_fraction, args.seed,
        )
    else:
        print(f"[prepare] Plain subset build: subsets={args.subsets} percentage={args.dataset_percentage}%")
        train_ds, val_ds = load_finetune_data(
            args.dataset_name, args.subsets, "train", args.seed, args.val_fraction,
            dataset_percentage=args.dataset_percentage,
        )

    print(f"[prepare] train rows={len(train_ds):,} val rows={len(val_ds):,}")

    os.makedirs(args.output_dir, exist_ok=True)
    DatasetDict({"train": train_ds, "validation": val_ds}).save_to_disk(args.output_dir)
    print(f"[prepare] Saved consolidated dataset to: {args.output_dir}")

    if not args.no_cleanup:
        cleanup_raw_hf_cache(args.dataset_name)

    du_bytes = sum(f.stat().st_size for f in __import__("pathlib").Path(args.output_dir).rglob("*") if f.is_file())
    print(f"[prepare] Final on-disk size of {args.output_dir}: {du_bytes / (1024**3):.2f}GB")


if __name__ == "__main__":
    main()
