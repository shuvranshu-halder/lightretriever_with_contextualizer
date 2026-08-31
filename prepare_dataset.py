#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Stage 1 of 2: build a single, disk-budget-constrained, self-contained
training dataset ONCE, then delete HF's raw download cache to avoid the
"parquet download + Arrow cache" double-storage problem.
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
    """Delete HF's raw download + intermediate Arrow-conversion cache for this dataset repo."""
    freed_bytes = 0
    slug = dataset_name.replace("/", "--")

    candidates = []
    hub_cache = getattr(hf_constants, "HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
    candidates += glob.glob(os.path.join(hub_cache, f"datasets--{slug}*"))

    datasets_cache = os.path.expanduser(os.environ.get("HF_DATASETS_CACHE", "~/.cache/huggingface/datasets"))
    candidates += glob.glob(os.path.join(datasets_cache, f"*{slug}*"))
    candidates += glob.glob(os.path.join(datasets_cache, "parquet", f"*{slug}*"))
    candidates += glob.glob(os.path.join(datasets_cache, "downloads", "*"))

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
    config = DataConfig()

    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, default=config.dataset_name)

    p.add_argument("--data_mixture_config", type=str, default=os.environ.get("DATA_MIXTURE_CONFIG", config.data_mixture_config))
    p.add_argument("--disk_budget_gb", type=float, default=os.environ.get("DISK_BUDGET_GB", config.disk_budget_gb))

    default_subsets = config.subsets if isinstance(config.subsets, list) else ["msmarco"]
    p.add_argument("--subsets", type=str, nargs="+", default=default_subsets)
    p.add_argument("--dataset_percentage", type=float, default=config.dataset_percentage)

    p.add_argument("--val_fraction", type=float, default=config.val_fraction)
    p.add_argument("--seed", type=int, default=config.seed)
    p.add_argument("--output_dir", type=str, default=os.environ.get("PREPARED_DATASET_DIR", "./data/prepared"))
    p.add_argument("--no_cleanup", action="store_true", help="Skip deleting the raw HF download cache afterward.")
    p.add_argument("--force", action="store_true", help="Rebuild even if --output_dir already has a prepared dataset.")
    args = p.parse_args()

    if os.path.exists(os.path.join(args.output_dir, "dataset_dict.json")) and not args.force:
        print(f"[prepare] {args.output_dir} already contains a prepared dataset. Skipping.")
        return

    # RESTORED BUDGET SELECTION ROUTING
    if args.data_mixture_config and args.disk_budget_gb and os.path.exists(args.data_mixture_config):
        print(f"[prepare] Budget-constrained build: {args.disk_budget_gb}GB via {args.data_mixture_config}")
        train_ds, val_ds = load_finetune_data_with_budget(
            args.dataset_name, args.data_mixture_config, args.disk_budget_gb, args.val_fraction, args.seed,
        )
    else:
        print(f"[prepare] Plain subset build fallback: subsets={args.subsets} percentage={args.dataset_percentage}%")
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
