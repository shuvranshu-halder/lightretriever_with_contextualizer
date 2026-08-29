#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Training configuration for the LightRetriever query-side Contextualizer.

Frozen components:
    - EmbeddingBag lookup table (cached via scripts/cache_emb_bag.ipynb)
    - Document LLM encoder (lightretriever-llama3.2-3b, LoRA merged)

Trainable component:
    - A small (default 3-layer) bidirectional Transformer "contextualizer"
      that sits on top of the frozen per-token embeddings and produces a
      contextualized query embedding via mean pooling.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    # Frozen document encoder (same checkpoint used to build the EmbeddingBag)
    doc_model_name_or_path: str = "lightretriever/lightretriever-llama3.2-3b"
    attn_implementation: str = "sdpa"  # per user's choice, instead of flash_attention_2

    # Frozen lookup table produced by scripts/cache_emb_bag.ipynb
    emb_bag_path: str = "llama3.2_3b.web_search_en.emb_bag.pt"

    # Contextualizer architecture
    num_layers: int = 3
    num_heads: int = 16          # hidden_size (3072) must be divisible by num_heads
    ffn_multiplier: int = 2      # FFN dim = ffn_multiplier * hidden_size (trimmed from usual 4x)
    dropout: float = 0.1
    max_query_len: int = 64      # max number of *query tokens* (excludes prompt), for learned pos-emb
    activation: str = "gelu"

    # Query prompt - MUST match the prompt used when the EmbeddingBag was constructed
    query_prompt: str = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "


@dataclass
class DataConfig:
    dataset_name: str = "lightretriever/lightretriever-finetune-data"
    # HF dataset "config name" (subset). Pass a list to interleave multiple subsets.
    subsets: list = field(default_factory=lambda: ["msmarco"])
    num_hard_negatives: int = 7     # sampled per query, in addition to in-batch negatives
    max_doc_len: int = 256
    max_query_len: int = 64         # tokens, excludes special tokens/prompt
    train_split: str = "train"
    val_fraction: float = 0.01      # carve out a small held-out slice for eval
    seed: int = 42


@dataclass
class TrainConfig:
    output_dir: str = "./outputs/contextualizer_llama3b"
    log_file: str = "train.log"

    per_device_train_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    num_train_epochs: float = 1.0
    max_steps: Optional[int] = None   # overrides num_train_epochs if set

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    temperature: float = 0.02

    logging_steps: int = 20
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3

    bf16: bool = True
    seed: int = 42
    num_workers: int = 4

    resume_from_checkpoint: Optional[str] = None
