#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Dataset loading for `lightretriever/lightretriever-finetune-data`.

Schema per row (confirmed on the HF dataset viewer):
    query_id:           str
    query:              str   (task-specific instruction is ALREADY baked into
                                the text, e.g. "Instruct: ...\nQuery: ...")
    positive_passages:  list[{"docid": str, "text": str, "title": str}]
    negative_passages:  list[{"docid": str, "text": str, "title": str}]  (hard negatives)

IMPORTANT: The dataset's `query` field already contains whatever task
instruction is relevant for that subset. Do NOT re-prepend
`ModelConfig.query_prompt` here - that prompt was only used once, to bake a
*generic* context into the frozen per-token lookup table itself
(scripts/cache_emb_bag.ipynb). Every individual query token is looked up
context-free from that table regardless of surrounding text, so any
instruction text in the dataset's `query` field is simply tokenized as
ordinary tokens and handed to the contextualizer like everything else.
"""
import random
from typing import Optional

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
from transformers.tokenization_utils import PreTrainedTokenizerBase


def format_passage(passage: dict) -> str:
    title = (passage.get("title") or "").strip()
    text = (passage.get("text") or "").strip()
    return f"{title} {text}".strip() if title else text


def load_finetune_data(dataset_name: str, subsets: list[str], split: str, seed: int, val_fraction: float):
    """Load one or more subsets of the dataset and concatenate them, then
    carve out a small deterministic validation slice."""
    all_ds = []
    for subset in subsets:
        ds = load_dataset(dataset_name, subset, split=split)
        all_ds.append(ds)
    full_ds = all_ds[0] if len(all_ds) == 1 else concatenate_datasets(all_ds)

    split_ds = full_ds.train_test_split(test_size=val_fraction, seed=seed)
    return split_ds["train"], split_ds["test"]


class RetrievalContrastiveDataset(Dataset):
    """Each item = one query + one sampled positive + up to `num_hard_negatives`
    sampled hard negatives. In-batch negatives are formed implicitly at
    collate/training time from other queries' documents in the same batch."""

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
            # Not enough hard negatives - use all available (batch still gets
            # in-batch negatives from other queries to compensate).
            negs = list(neg_list)

        return {
            "query": query,
            "pos_text": format_passage(pos),
            "neg_texts": [format_passage(n) for n in negs],
        }


class Collator:
    """Tokenizes queries (for the contextualizer) and documents (for the
    frozen LLM encoder), and tracks which document index is the true
    positive for each query so the loss function can build its label vector.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_query_len: int,
        max_doc_len: int,
    ):
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

    def __call__(self, batch: list[dict]) -> dict:
        queries = [ex["query"] for ex in batch]

        doc_texts: list[str] = []
        positive_indices: list[int] = []
        for ex in batch:
            positive_indices.append(len(doc_texts))  # index of this query's positive
            doc_texts.append(ex["pos_text"])
            doc_texts.extend(ex["neg_texts"])

        # --- Query tokenization: matches tokenize_nonctx_qry_emb_bag in
        # scripts/asymmetric_dense_infer.ipynb (add_special_tokens=False),
        # since every vocab id (including bos/eos) already has its own row
        # in the frozen lookup table.
        query_enc = self.tokenizer(
            queries,
            max_length=self.max_query_len,
            padding="longest",
            truncation=True,
            add_special_tokens=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # --- Document tokenization: matches the doc-side example in
        # scripts/asymmetric_dense_infer.ipynb (add_special_tokens=True).
        doc_enc = self.tokenizer(
            doc_texts,
            max_length=self.max_doc_len,
            padding="longest",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "query_input_ids": query_enc["input_ids"],
            "query_attention_mask": query_enc["attention_mask"],
            "doc_input_ids": doc_enc["input_ids"],
            "doc_attention_mask": doc_enc["attention_mask"],
            "positive_indices": torch.tensor(positive_indices, dtype=torch.long),
        }
