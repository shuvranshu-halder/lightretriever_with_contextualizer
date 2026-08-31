#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Train the LightRetriever query-side Contextualizer on top of a FROZEN
EmbeddingBag lookup table (Llama-3.2-3B), using a FROZEN document LLM
encoder for the passage side.

Only the contextualizer (+ its positional embeddings) receives gradients.

Supports DDP (multi-GPU, one process per GPU). Launch with torchrun:

    torchrun --nproc_per_node=4 train.py \
        --emb_bag_path llama3.2_3b.web_search_en.emb_bag.pt \
        --doc_model_name_or_path lightretriever/lightretriever-llama3.2-3b \
        --attn_implementation sdpa \
        --subsets msmarco \
        --output_dir ./outputs/contextualizer_llama3b \
        --per_device_train_batch_size 32 \
        --num_train_epochs 1

Single GPU still works exactly as before (just `python train.py ...`,
no torchrun needed - the script detects it's not in a distributed launch).

Resume:
    torchrun --nproc_per_node=4 train.py ... \
        --resume_from_checkpoint ./outputs/contextualizer_llama3b/checkpoint-last
"""
import os
import sys
import time
import json
import glob
import shutil
import logging
import argparse
from dataclasses import asdict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, PeftModel

from config import ModelConfig, DataConfig, TrainConfig
from seeding import set_seed, seed_worker
from contextualizer import QueryContextualizer
from datasets import load_from_disk
from dataset import load_finetune_data, load_finetune_data_with_budget, RetrievalContrastiveDataset, Collator


# --------------------------------------------------------------------------- #
# Distributed setup
# --------------------------------------------------------------------------- #
def setup_distributed():
    """Detects a torchrun launch via env vars set by torchrun (RANK, WORLD_SIZE,
    LOCAL_RANK). Falls back to single-process/single-GPU if not present."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_distributed = True
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        is_distributed = False
    return rank, world_size, local_rank, device, is_distributed


def is_main_process(rank: int) -> bool:
    return rank == 0


def reduce_mean(value: float, device: torch.device, distributed: bool, world_size: int) -> float:
    """Average a python float metric across all ranks."""
    if not distributed:
        return value
    t = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / world_size).item()


# --------------------------------------------------------------------------- #
# Logging setup: rank 0 writes to console + log file; other ranks stay quiet
# (only warnings/errors to console) to avoid interleaved/corrupted log files.
# --------------------------------------------------------------------------- #
def setup_logging(output_dir: str, log_file: str, rank: int) -> logging.Logger:
    logger = logging.getLogger("contextualizer_train")
    logger.setLevel(logging.INFO if is_main_process(rank) else logging.WARNING)
    logger.handlers.clear()

    fmt = logging.Formatter(f"%(asctime)s | rank{rank} | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    if is_main_process(rank):
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, log_file)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if is_main_process(rank):
        logger.info(f"Logging to console and to: {os.path.join(output_dir, log_file)}")
    return logger


# --------------------------------------------------------------------------- #
# Frozen document encoder (identical setup to scripts/asymmetric_dense_infer.ipynb)
# Each rank loads its OWN copy onto its OWN GPU - it's frozen/eval-only, so
# there's no need (and no way) to DDP-wrap it; every rank just needs a local
# copy to encode its own shard of documents.
# --------------------------------------------------------------------------- #
def load_frozen_doc_model(model_name_or_path: str, attn_implementation: str, device: torch.device):
    config = LoraConfig.from_pretrained(model_name_or_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,  # "sdpa" per user's choice
        device_map=device,
    )
    hf_model = PeftModel.from_pretrained(base_model, model_name_or_path, config=config)
    hf_model = hf_model.merge_and_unload()
    hf_model.eval()
    for p in hf_model.parameters():
        p.requires_grad_(False)
    return hf_model


def lasttoken_pooling(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Verbatim logic from scripts/asymmetric_dense_infer.ipynb."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden[:, -1]
    sequence_lengths = attention_mask.sum(dim=1)
    last_token_indices = sequence_lengths - 1
    return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), last_token_indices]


@torch.no_grad()
def encode_documents(doc_model, input_ids, attention_mask) -> torch.Tensor:
    lm_out = doc_model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
        use_cache=False,
        output_hidden_states=False,
    )
    emb = lasttoken_pooling(lm_out.last_hidden_state, attention_mask)
    return F.normalize(emb, p=2, dim=-1)


# --------------------------------------------------------------------------- #
# Symmetric (bidirectional) InfoNCE loss
# NOTE: under DDP, each rank only sees its own local batch's documents as
# negatives (this loss does NOT gather documents across ranks). Effective
# in-batch negative pool per step = per_device_train_batch_size * (1+k),
# same as single-GPU, just parallelized across more queries/sec, not a
# larger negative pool. See note below train() if you want cross-rank
# negative gathering instead.
# --------------------------------------------------------------------------- #
def symmetric_infonce_loss(query_emb: torch.Tensor, doc_emb: torch.Tensor,
                            positive_indices: torch.Tensor, temperature: float):
    scores = query_emb @ doc_emb.T / temperature   # [B, N]

    loss_q2d = F.cross_entropy(scores, positive_indices)

    pos_doc_scores = scores.T[positive_indices]     # [B, B]
    targets_d2q = torch.arange(pos_doc_scores.size(0), device=scores.device)
    loss_d2q = F.cross_entropy(pos_doc_scores, targets_d2q)

    loss = (loss_q2d + loss_d2q) / 2.0

    with torch.no_grad():
        acc_q2d = (scores.argmax(dim=1) == positive_indices).float().mean().item()
    return loss, acc_q2d


# --------------------------------------------------------------------------- #
# Checkpointing (rank 0 only)
# --------------------------------------------------------------------------- #
def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def save_checkpoint(output_dir, step, model, optimizer, scheduler, best_metric, save_total_limit, logger, rank):
    if not is_main_process(rank):
        return
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save({
        "step": step,
        "model_state_dict": unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_metric": best_metric,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }, os.path.join(ckpt_dir, "training_state.pt"))

    latest_dir = os.path.join(output_dir, "checkpoint-last")
    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)
    shutil.copytree(ckpt_dir, latest_dir)

    logger.info(f"Saved checkpoint at step {step} -> {ckpt_dir}")

    numbered = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda p: int(p.split("-")[-1]) if p.split("-")[-1].isdigit() else -1,
    )
    numbered = [p for p in numbered if os.path.basename(p) != "checkpoint-last"]
    while len(numbered) > save_total_limit:
        to_remove = numbered.pop(0)
        shutil.rmtree(to_remove, ignore_errors=True)
        logger.info(f"Removed old checkpoint: {to_remove}")


def load_checkpoint(ckpt_dir, model, optimizer, scheduler, logger, device):
    state = torch.load(os.path.join(ckpt_dir, "training_state.pt"), map_location=device)
    unwrap(model).load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    torch.set_rng_state(state["rng_state"]["torch"].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state["rng_state"]["cuda"]])
    logger.info(f"Resumed from {ckpt_dir} at step {state['step']}")
    return state["step"], state.get("best_metric", -1.0)


# --------------------------------------------------------------------------- #
# Evaluation - every rank evaluates its own shard, metrics are all-reduced
# (averaged) so the logged number reflects the full validation set.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, doc_model, val_loader, device, temperature, logger, distributed, world_size, max_batches: int = 50):
    model.eval()
    total_acc, total_loss, n = 0.0, 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        query_emb = model(
            batch["query_input_ids"].to(device),
            batch["query_attention_mask"].to(device),
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            doc_emb = encode_documents(
                doc_model,
                batch["doc_input_ids"].to(device),
                batch["doc_attention_mask"].to(device),
            )
        loss, acc = symmetric_infonce_loss(
            query_emb, doc_emb.float(), batch["positive_indices"].to(device), temperature
        )
        total_loss += loss.item()
        total_acc += acc
        n += 1
    model.train()
    avg_loss = total_loss / max(n, 1)
    avg_acc = total_acc / max(n, 1)

    avg_loss = reduce_mean(avg_loss, device, distributed, world_size)
    avg_acc = reduce_mean(avg_acc, device, distributed, world_size)

    logger.info(f"[EVAL] loss={avg_loss:.4f} acc@1={avg_acc:.4f} (over {n} local batches/rank)")
    return avg_acc


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--doc_model_name_or_path", type=str, default=ModelConfig.doc_model_name_or_path)
    p.add_argument("--attn_implementation", type=str, default=ModelConfig.attn_implementation)
    p.add_argument("--emb_bag_path", type=str, required=True)
    p.add_argument("--num_layers", type=int, default=ModelConfig.num_layers)
    p.add_argument("--num_heads", type=int, default=ModelConfig.num_heads)

    p.add_argument("--prepared_dataset_dir", type=str, default=None,
                    help="Path built by prepare_dataset.py. If set, skips all HF download/mixture logic "
                         "below and loads this local, pre-built dataset directly (no network needed).")
    p.add_argument("--subsets", type=str, nargs="+", default=DataConfig.subsets)
    p.add_argument("--dataset_percentage", type=float, default=DataConfig.dataset_percentage,
                    help="Keep only this percent of each subset's rows, e.g. 10 = 10%%")
    p.add_argument("--data_mixture_config", type=str, default=None,
                    help="Path to exp-m.json (cloned lightretriever repo: config/data/exp-m.json). "
                         "If set together with --disk_budget_gb, this overrides --subsets/--dataset_percentage "
                         "and builds a budget-constrained set of subsets proportional to the official domain_weights.")
    p.add_argument("--disk_budget_gb", type=float, default=None,
                    help="Hard cap (in GB) on total downloaded parquet data, allocated across subsets "
                         "proportional to domain_weights from --data_mixture_config.")
    p.add_argument("--num_hard_negatives", type=int, default=DataConfig.num_hard_negatives)
    p.add_argument("--max_doc_len", type=int, default=DataConfig.max_doc_len)
    p.add_argument("--max_query_len", type=int, default=DataConfig.max_query_len)

    p.add_argument("--output_dir", type=str, default=TrainConfig.output_dir)
    p.add_argument("--per_device_train_batch_size", type=int, default=TrainConfig.per_device_train_batch_size)
    p.add_argument("--gradient_accumulation_steps", type=int, default=TrainConfig.gradient_accumulation_steps)
    p.add_argument("--num_train_epochs", type=float, default=TrainConfig.num_train_epochs)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=TrainConfig.learning_rate)
    p.add_argument("--temperature", type=float, default=TrainConfig.temperature)
    p.add_argument("--logging_steps", type=int, default=TrainConfig.logging_steps)
    p.add_argument("--eval_steps", type=int, default=TrainConfig.eval_steps)
    p.add_argument("--save_steps", type=int, default=TrainConfig.save_steps)
    p.add_argument("--save_total_limit", type=int, default=TrainConfig.save_total_limit)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank, device, distributed = setup_distributed()

    model_cfg = ModelConfig(
        doc_model_name_or_path=args.doc_model_name_or_path,
        attn_implementation=args.attn_implementation,
        emb_bag_path=args.emb_bag_path,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_query_len=args.max_query_len,
    )
    data_cfg = DataConfig(
        subsets=args.subsets,
        dataset_percentage=args.dataset_percentage,
        data_mixture_config=args.data_mixture_config,
        disk_budget_gb=args.disk_budget_gb,
        num_hard_negatives=args.num_hard_negatives,
        max_doc_len=args.max_doc_len,
        max_query_len=args.max_query_len,
        seed=args.seed,
    )
    train_cfg = TrainConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    logger = setup_logging(train_cfg.output_dir, train_cfg.log_file, rank)
    if is_main_process(rank):
        logger.info(f"Distributed: {distributed} | world_size={world_size}")
        logger.info("===== Config =====")
        logger.info(json.dumps({
            "model": asdict(model_cfg), "data": asdict(data_cfg), "train": asdict(train_cfg)
        }, indent=2, default=str))

    # Same seed on every rank -> identical model init before DDP broadcast,
    # identical dropout patterns per-rank-position. DistributedSampler
    # handles giving each rank a different data shard despite the same seed.
    set_seed(train_cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.doc_model_name_or_path)

    if is_main_process(rank):
        logger.info(f"Loading frozen document encoder: {model_cfg.doc_model_name_or_path} "
                    f"(attn_implementation={model_cfg.attn_implementation})")
    doc_model = load_frozen_doc_model(model_cfg.doc_model_name_or_path, model_cfg.attn_implementation, device)

    if is_main_process(rank):
        logger.info(f"Loading frozen EmbeddingBag weights from: {model_cfg.emb_bag_path}")
    emb_bag_weight = torch.load(model_cfg.emb_bag_path, map_location="cpu")

    model = QueryContextualizer(model_cfg, emb_bag_weight, pad_token_id=tokenizer.pad_token_id)
    model.to(device=device, dtype=torch.float32)

    if distributed:
        # find_unused_parameters=False: every trainable param participates in
        # every forward pass here, so keep this False for speed.
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    n_trainable = sum(p.numel() for p in unwrap(model).trainable_parameters())
    n_frozen = sum(p.numel() for p in unwrap(model).parameters() if not p.requires_grad)
    if is_main_process(rank):
        logger.info(f"Contextualizer trainable params: {n_trainable:,} | frozen (lookup table) params: {n_frozen:,}")

    if is_main_process(rank):
        logger.info(f"Loading dataset subsets: {data_cfg.subsets}")
    if args.prepared_dataset_dir:
        if is_main_process(rank):
            logger.info(f"Loading pre-built dataset from: {args.prepared_dataset_dir} (no network)")
        dsd = load_from_disk(args.prepared_dataset_dir)
        train_hf, val_hf = dsd["train"], dsd["validation"]
    elif data_cfg.data_mixture_config and data_cfg.disk_budget_gb:
        if is_main_process(rank):
            logger.info(f"Using budget-constrained mixture loading: "
                        f"config={data_cfg.data_mixture_config} budget={data_cfg.disk_budget_gb}GB")
        train_hf, val_hf = load_finetune_data_with_budget(
            data_cfg.dataset_name, data_cfg.data_mixture_config, data_cfg.disk_budget_gb,
            data_cfg.val_fraction, data_cfg.seed,
        )
    else:
        train_hf, val_hf = load_finetune_data(
            data_cfg.dataset_name, data_cfg.subsets, data_cfg.train_split, data_cfg.seed, data_cfg.val_fraction,
            dataset_percentage=data_cfg.dataset_percentage,
        )
    if is_main_process(rank):
        logger.info(f"Train examples: {len(train_hf):,} | Val examples: {len(val_hf):,}")

    train_ds = RetrievalContrastiveDataset(train_hf, data_cfg.num_hard_negatives, data_cfg.seed + rank)
    val_ds = RetrievalContrastiveDataset(val_hf, data_cfg.num_hard_negatives, data_cfg.seed + 1 + rank)

    collator = Collator(tokenizer, data_cfg.max_query_len, data_cfg.max_doc_len)

    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=train_cfg.seed, drop_last=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        train_loader = DataLoader(
            train_ds, batch_size=train_cfg.per_device_train_batch_size, sampler=train_sampler,
            collate_fn=collator, num_workers=train_cfg.num_workers, worker_init_fn=seed_worker,
            drop_last=True, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=train_cfg.per_device_train_batch_size, sampler=val_sampler,
            collate_fn=collator, num_workers=train_cfg.num_workers, pin_memory=True,
        )
    else:
        train_sampler = None
        g = torch.Generator()
        g.manual_seed(train_cfg.seed)
        train_loader = DataLoader(
            train_ds, batch_size=train_cfg.per_device_train_batch_size, shuffle=True,
            collate_fn=collator, num_workers=train_cfg.num_workers, worker_init_fn=seed_worker,
            generator=g, drop_last=True, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=train_cfg.per_device_train_batch_size, shuffle=False,
            collate_fn=collator, num_workers=train_cfg.num_workers, pin_memory=True,
        )

    steps_per_epoch = len(train_loader) // train_cfg.gradient_accumulation_steps
    total_steps = train_cfg.max_steps or int(steps_per_epoch * train_cfg.num_train_epochs)
    warmup_steps = int(total_steps * train_cfg.warmup_ratio)

    optimizer = torch.optim.AdamW(
        unwrap(model).trainable_parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if is_main_process(rank):
        logger.info(f"Total optimizer steps: {total_steps} | warmup steps: {warmup_steps} "
                    f"| world_size={world_size} | effective batch size="
                    f"{train_cfg.per_device_train_batch_size * world_size * train_cfg.gradient_accumulation_steps}")

    global_step = 0
    best_metric = -1.0
    if train_cfg.resume_from_checkpoint:
        global_step, best_metric = load_checkpoint(
            train_cfg.resume_from_checkpoint, model, optimizer, scheduler, logger, device
        )
        if distributed:
            dist.barrier()

    model.train()
    running_loss, running_acc = 0.0, 0.0
    t0 = time.time()
    step_in_accum = 0
    epoch = 0

    data_iter = iter(train_loader)
    if distributed:
        train_sampler.set_epoch(epoch)

    while global_step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if distributed:
                train_sampler.set_epoch(epoch)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        query_input_ids = batch["query_input_ids"].to(device)
        query_attention_mask = batch["query_attention_mask"].to(device)
        doc_input_ids = batch["doc_input_ids"].to(device)
        doc_attention_mask = batch["doc_attention_mask"].to(device)
        positive_indices = batch["positive_indices"].to(device)

        step_in_accum += 1
        is_sync_step = (step_in_accum == train_cfg.gradient_accumulation_steps)

        # Skip DDP gradient all-reduce on non-final accumulation micro-steps.
        sync_ctx = model.no_sync() if (distributed and not is_sync_step) else _nullcontext()
        with sync_ctx:
            query_emb = model(query_input_ids, query_attention_mask)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                doc_emb = encode_documents(doc_model, doc_input_ids, doc_attention_mask)
            doc_emb = doc_emb.float()

            loss, acc = symmetric_infonce_loss(query_emb, doc_emb, positive_indices, train_cfg.temperature)
            (loss / train_cfg.gradient_accumulation_steps).backward()

        running_loss += loss.item()
        running_acc += acc

        if is_sync_step:
            grad_norm = torch.nn.utils.clip_grad_norm_(unwrap(model).trainable_parameters(), train_cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step_in_accum = 0
            global_step += 1

            if global_step % train_cfg.logging_steps == 0:
                elapsed = time.time() - t0
                avg_loss = reduce_mean(
                    running_loss / (train_cfg.logging_steps * train_cfg.gradient_accumulation_steps),
                    device, distributed, world_size,
                )
                avg_acc = reduce_mean(
                    running_acc / (train_cfg.logging_steps * train_cfg.gradient_accumulation_steps),
                    device, distributed, world_size,
                )
                lr = scheduler.get_last_lr()[0]
                if is_main_process(rank):
                    logger.info(
                        f"step={global_step}/{total_steps} loss={avg_loss:.4f} acc@1={avg_acc:.4f} "
                        f"lr={lr:.2e} grad_norm={grad_norm:.3f} sec/step={elapsed / train_cfg.logging_steps:.2f}"
                    )
                running_loss, running_acc = 0.0, 0.0
                t0 = time.time()

            if global_step % train_cfg.eval_steps == 0:
                metric = evaluate(model, doc_model, val_loader, device, train_cfg.temperature, logger, distributed, world_size)
                if metric > best_metric:
                    best_metric = metric
                    if is_main_process(rank):
                        logger.info(f"New best acc@1={best_metric:.4f} at step {global_step}")

            if global_step % train_cfg.save_steps == 0:
                save_checkpoint(
                    train_cfg.output_dir, global_step, model, optimizer, scheduler,
                    best_metric, train_cfg.save_total_limit, logger, rank
                )
                if distributed:
                    dist.barrier()

    save_checkpoint(train_cfg.output_dir, global_step, model, optimizer, scheduler,
                     best_metric, train_cfg.save_total_limit, logger, rank)
    if is_main_process(rank):
        logger.info("Training complete.")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
