#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Dataset loading for `lightretriever/lightretriever-finetune-data`.
"""
import json
import math
import random
from typing import Optional

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import HfApi, hf_hub_url
from transformers.tokenization_utils import PreTrainedTokenizerBase


def format_passage(passage: dict) -> str:
    title = (passage.get("title") or "").strip()
    text = (passage.get("text") or "").strip()
    return f"{title} {text}".strip() if title else text


def get_repo_file_sizes(dataset_name: str) -> dict[str, int]:
    """Fetch {relative_filepath: size_in_bytes} for every file in the dataset
    repo, in a single API call."""
    api = HfApi()
    info = api.dataset_info(dataset_name, files_metadata=True)
    return {s.rfilename: (s.size or 0) for s in info.siblings if s.rfilename.endswith(".parquet")}


def get_shard_files_for_subset(dataset_name: str, subset: str) -> list[str]:
    """List every parquet shard file belonging to one subset (config) of the
    dataset repo, e.g. files under 'msmarco/train-00000-of-00032.parquet'."""
    api = HfApi()
    all_files = api.list_repo_files(dataset_name, repo_type="dataset")
    prefix = f"{subset}/"
    shard_files = sorted(f for f in all_files if f.startswith(prefix) and f.endswith(".parquet"))
    if not shard_files:
        raise ValueError(
            f"No parquet shard files found under '{prefix}' in dataset repo '{dataset_name}'. "
            f"Check the subset name against the dataset's Files tab on the Hub."
        )
    return shard_files


def select_shard_subset(shard_files: list[str], percentage: float, seed: int) -> list[str]:
    """Randomly (but deterministically) keep `percentage`% of the shard files."""
    n_total = len(shard_files)
    n_keep = max(1, math.ceil(n_total * percentage / 100.0))
    rng = random.Random(seed)
    return sorted(rng.sample(shard_files, n_keep))


def load_domain_weights(mixture_config_path: str) -> dict[str, float]:
    """Load the official {subset: domain_weight} mixture from a config like
    lightretriever/config/data/exp-m.json (cloned from the repo)."""
    with open(mixture_config_path) as f:
        cfg = json.load(f)
    weights = cfg["domain_weights"]
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}  # re-normalize just in case


def plan_budget_shard_selection(
    dataset_name: str,
    domain_weights: dict[str, float],
    total_budget_gb: float,
    seed: int,
) -> dict[str, list[str]]:
    """Decide which parquet shard files to download for each subset so that
    the TOTAL download stays within `total_budget_gb`."""
    all_sizes = get_repo_file_sizes(dataset_name)  # {filepath: bytes}

    subset_files: dict[str, list[tuple[str, int]]] = {}
    subset_total_bytes: dict[str, int] = {}
    for subset in domain_weights:
        prefix = f"{subset}/"
        files = sorted((fp, sz) for fp, sz in all_sizes.items() if fp.startswith(prefix))
        if not files:
            raise ValueError(f"No parquet files found under '{prefix}' in '{dataset_name}'.")
        subset_files[subset] = files
        subset_total_bytes[subset] = sum(sz for _, sz in files)

    budget_bytes = int(total_budget_gb * (1024 ** 3))
    remaining_weights = dict(domain_weights)
    remaining_budget = budget_bytes
    target_bytes: dict[str, int] = {}

    for _ in range(len(domain_weights) + 1):
        if not remaining_weights:
            break
        w_sum = sum(remaining_weights.values())
        newly_capped = {}
        still_open = {}
        for subset, w in remaining_weights.items():
            proposed = int(remaining_budget * (w / w_sum))
            available = subset_total_bytes[subset]
            if proposed >= available:
                newly_capped[subset] = available
            else:
                still_open[subset] = w
        for subset, cap in newly_capped.items():
            target_bytes[subset] = cap
            remaining_budget -= cap
        remaining_weights = still_open
        if not newly_capped:
            for subset, w in still_open.items():
                target_bytes[subset] = int(remaining_budget * (w / w_sum))
            break

    rng = random.Random(seed)
    selected: dict[str, list[str]] = {}
    achieved_total = 0
    for subset, files in subset_files.items():
        avg_shard_size = subset_total_bytes[subset] / len(files)
        n_keep = max(1, min(len(files), round(target_bytes.get(subset, 0) / avg_shard_size)))
        chosen = sorted(rng.sample(files, n_keep), key=lambda x: x[0])
        selected[subset] = [fp for fp, _ in chosen]
        achieved_total += sum(sz for _, sz in chosen)

    print(f"[budget planner] requested={total_budget_gb:.2f}GB "
          f"achieved~={achieved_total / (1024**3):.2f}GB across {len(selected)} subsets")
    for subset, files in selected.items():
        kept_bytes = sum(sz for fp, sz in subset_files[subset] if fp in files)
        print(f"  {subset}: {len(files)}/{len(subset_files[subset])} shards "
              f"(~{kept_bytes / (1024**3):.3f}GB, weight={domain_weights[subset]:.4f})")

    return selected


def load_finetune_data_with_budget(
    dataset_name: str,
    mixture_config_path: str,
    total_budget_gb: float,
    val_fraction: float,
    seed: int,
):
    """Build the training set within a hard disk budget, allocated proportionally."""
    domain_weights = load_domain_weights(mixture_config_path)
    selected_files = plan_budget_shard_selection(dataset_name, domain_weights, total_budget_gb, seed)

    all_ds = []
    for subset, files in selected_files.items():
        urls = [hf_hub_url(dataset_name, filename=f, repo_type="dataset") for f in files]
        # CHANGE: Added keep_in_memory=True to keep Arrow tables in RAM
        print(f"\n" + "="*60)
        print(f"[PROCESS] Generating splits for dataset subset: {subset.upper()}")
        print(f"Loading {len(files)} shards into RAM...")
        print("="*60 + "\n")
        
        ds = load_dataset("parquet", data_files=urls, split="train", keep_in_memory=True)
        all_ds.append(ds)

    # CHANGE: Ensure concatenation and shuffling happen strictly inside RAM
    full_ds = concatenate_datasets(all_ds)
    full_ds = full_ds.shuffle(seed=seed, keep_in_memory=True)
    split_ds = full_ds.train_test_split(test_size=val_fraction, seed=seed, keep_in_memory=True)
    return split_ds["train"], split_ds["test"]



def load_finetune_data(
    dataset_name: str,
    subsets: list[str],
    split: str,
    seed: int,
    val_fraction: float,
    dataset_percentage: float = 100.0,
):
    """Load one or more subsets via file-shard downsampling."""
    assert 0 < dataset_percentage <= 100.0, "dataset_percentage must be in (0, 100]"
    assert split == "train", "This dataset only exposes a 'train' split per subset."

    all_ds = []
    for subset in subsets:
        shard_files = get_shard_files_for_subset(dataset_name, subset)
        if dataset_percentage < 100.0:
            kept = select_shard_subset(shard_files, dataset_percentage, seed)
        else:
            kept = shard_files
        urls = [hf_hub_url(dataset_name, filename=f, repo_type="dataset") for f in kept]
        ds = load_dataset("parquet", data_files=urls, split="train")
        all_ds.append(ds)

    full_ds = all_ds[0] if len(all_ds) == 1 else concatenate_datasets(all_ds)
    split_ds = full_ds.train_test_split(test_size=val_fraction, seed=seed)
    return split_ds["train"], split_ds["test"]


class RetrievalContrastiveDataset(Dataset):
    def __init__(self, hf_dataset, num_hard_negatives: int, seed: int):
        self.ds = hf_dataset
        self.num_hard_negatives = num_hard_negatives
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        row = self.ds[idx]
        query: str = row["query"]
        pos_list = row["positive_passages"]
        neg_list = row["negative_passages"]

        pos = self.rng.choice(pos_list)
        if len(neg_list) >= self.num_hard_negatives:
            negs = self.rng.sample(neg_list, self.num_hard_negatives)
        else:
            negs = list(neg_list)

        return {
            "query": query,
            "pos_text": format_passage(pos),
            "neg_texts": [format_passage(n) for n in negs],
        }


class Collator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_query_len: int, max_doc_len: int):
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

    def __call__(self, batch: list[dict]) -> dict:
        queries = [ex["query"] for ex in batch]
        doc_texts: list[str] = []
        positive_indices: list[int] = []
        for ex in batch:
            positive_indices.append(len(doc_texts))
            doc_texts.append(ex["pos_text"])
            doc_texts.extend(ex["neg_texts"])

        query_enc = self.tokenizer(
            queries, max_length=self.max_query_len, padding="longest",
            truncation=True, add_special_tokens=False, return_attention_mask=True, return_tensors="pt",
        )
        doc_enc = self.tokenizer(
            doc_texts, max_length=self.max_doc_len, padding="longest",
            truncation=True, add_special_tokens=True, return_attention_mask=True, return_tensors="pt",
        )

        return {
            "query_input_ids": query_enc["input_ids"],
            "query_attention_mask": query_enc["attention_mask"],
            "doc_input_ids": doc_enc["input_ids"],
            "doc_attention_mask": doc_enc["attention_mask"],
            "positive_indices": torch.tensor(positive_indices, dtype=torch.long),
        }
