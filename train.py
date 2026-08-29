#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Train the LightRetriever query-side Contextualizer on top of a FROZEN
EmbeddingBag lookup table (Llama-3.2-3B), using a FROZEN document LLM
encoder for the passage side.

Only the contextualizer (+ its positional embeddings) receives gradients.

Usage:
    python train.py \
        --emb_bag_path llama3.2_3b.web_search_en.emb_bag.pt \
        --doc_model_name_or_path lightretriever/lightretriever-llama3.2-3b \
        --subsets msmarco nq hotpotqa \
        --output_dir ./outputs/contextualizer_llama3b \
        --per_device_train_batch_size 64 \
        --num_train_epochs 1

Resume:
    python train.py ... --resume_from_checkpoint ./outputs/contextualizer_llama3b/checkpoint-last
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, PeftModel

from config import ModelConfig, DataConfig, TrainConfig
from seeding import set_seed, seed_worker
from contextualizer import QueryContextualizer
from dataset import load_finetune_data, RetrievalContrastiveDataset, Collator


# --------------------------------------------------------------------------- #
# Logging setup: everything goes to BOTH console and a log file.
# --------------------------------------------------------------------------- #
def setup_logging(output_dir: str, log_file: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_file)

    logger = logging.getLogger("contextualizer_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    logger.info(f"Logging to console and to: {log_path}")
    return logger


# --------------------------------------------------------------------------- #
# Frozen document encoder (identical setup to scripts/asymmetric_dense_infer.ipynb)
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
# --------------------------------------------------------------------------- #
def symmetric_infonce_loss(query_emb: torch.Tensor, doc_emb: torch.Tensor,
                            positive_indices: torch.Tensor, temperature: float):
    """
    query_emb: [B, H] normalized
    doc_emb:   [N, H] normalized  (N = sum over batch of (1 pos + k negs))
    positive_indices: [B] -> index into doc_emb of each query's true positive
    """
    scores = query_emb @ doc_emb.T / temperature   # [B, N]

    # Query -> Document direction
    loss_q2d = F.cross_entropy(scores, positive_indices)

    # Document -> Query direction (only defined for the positive docs)
    pos_doc_scores = scores.T[positive_indices]     # [B, B]  (row j = doc of query j, col i = query i)
    targets_d2q = torch.arange(pos_doc_scores.size(0), device=scores.device)
    loss_d2q = F.cross_entropy(pos_doc_scores, targets_d2q)

    loss = (loss_q2d + loss_d2q) / 2.0

    with torch.no_grad():
        acc_q2d = (scores.argmax(dim=1) == positive_indices).float().mean().item()
    return loss, acc_q2d


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(output_dir, step, model, optimizer, scheduler, best_metric, save_total_limit, logger):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),          # contextualizer only (frozen doc/table excluded via requires_grad, but state_dict includes buffers - fine, they're identical/frozen anyway)
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_metric": best_metric,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
        },
    }, os.path.join(ckpt_dir, "training_state.pt"))

    # Convenience symlink/copy for easy resume
    latest_dir = os.path.join(output_dir, "checkpoint-last")
    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)
    shutil.copytree(ckpt_dir, latest_dir)

    logger.info(f"Saved checkpoint at step {step} -> {ckpt_dir}")

    # Enforce save_total_limit (keep only the N most recent numbered checkpoints; "checkpoint-last" is always kept)
    numbered = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda p: int(p.split("-")[-1]) if p.split("-")[-1].isdigit() else -1,
    )
    numbered = [p for p in numbered if os.path.basename(p) != "checkpoint-last"]
    while len(numbered) > save_total_limit:
        to_remove = numbered.pop(0)
        shutil.rmtree(to_remove, ignore_errors=True)
        logger.info(f"Removed old checkpoint: {to_remove}")


def load_checkpoint(ckpt_dir, model, optimizer, scheduler, logger):
    state = torch.load(os.path.join(ckpt_dir, "training_state.pt"), map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    torch.set_rng_state(state["rng_state"]["torch"])
    torch.cuda.set_rng_state_all(state["rng_state"]["cuda"])
    logger.info(f"Resumed from {ckpt_dir} at step {state['step']}")
    return state["step"], state.get("best_metric", -1.0)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, doc_model, val_loader, device, temperature, logger, max_batches: int = 50):
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
    logger.info(f"[EVAL] loss={avg_loss:.4f} acc@1={avg_acc:.4f} (over {n} batches)")
    return avg_acc


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--doc_model_name_or_path", type=str, default=ModelConfig.doc_model_name_or_path)
    p.add_argument("--attn_implementation", type=str, default=ModelConfig.attn_implementation)
    p.add_argument("--emb_bag_path", type=str, required=True)
    p.add_argument("--num_layers", type=int, default=ModelConfig.num_layers)
    p.add_argument("--num_heads", type=int, default=ModelConfig.num_heads)

    p.add_argument("--subsets", type=str, nargs="+", default=DataConfig.subsets)
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

    logger = setup_logging(train_cfg.output_dir, train_cfg.log_file)
    logger.info("===== Config =====")
    logger.info(json.dumps({
        "model": asdict(model_cfg), "data": asdict(data_cfg), "train": asdict(train_cfg)
    }, indent=2, default=str))

    set_seed(train_cfg.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- Tokenizer (shared for query + doc, same model family) ----
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.doc_model_name_or_path)

    # ---- Frozen document encoder ----
    logger.info(f"Loading frozen document encoder: {model_cfg.doc_model_name_or_path} "
                f"(attn_implementation={model_cfg.attn_implementation})")
    doc_model = load_frozen_doc_model(model_cfg.doc_model_name_or_path, model_cfg.attn_implementation, device)

    # ---- Frozen lookup table ----
    logger.info(f"Loading frozen EmbeddingBag weights from: {model_cfg.emb_bag_path}")
    emb_bag_weight = torch.load(model_cfg.emb_bag_path, map_location="cpu")

    # ---- Trainable contextualizer ----
    model = QueryContextualizer(model_cfg, emb_bag_weight, pad_token_id=tokenizer.pad_token_id)
    model.to(device=device, dtype=torch.float32)  # keep trainable params in fp32 master weights; autocast handles bf16 compute
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(f"Contextualizer trainable params: {n_trainable:,} | frozen (lookup table) params: {n_frozen:,}")

    # ---- Data ----
    logger.info(f"Loading dataset subsets: {data_cfg.subsets}")
    train_hf, val_hf = load_finetune_data(
        data_cfg.dataset_name, data_cfg.subsets, data_cfg.train_split, data_cfg.seed, data_cfg.val_fraction
    )
    logger.info(f"Train examples: {len(train_hf):,} | Val examples: {len(val_hf):,}")

    train_ds = RetrievalContrastiveDataset(train_hf, data_cfg.num_hard_negatives, data_cfg.seed)
    val_ds = RetrievalContrastiveDataset(val_hf, data_cfg.num_hard_negatives, data_cfg.seed + 1)

    collator = Collator(tokenizer, data_cfg.max_query_len, data_cfg.max_doc_len)

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

    # ---- Optimizer / scheduler ----
    steps_per_epoch = len(train_loader) // train_cfg.gradient_accumulation_steps
    total_steps = train_cfg.max_steps or int(steps_per_epoch * train_cfg.num_train_epochs)
    warmup_steps = int(total_steps * train_cfg.warmup_ratio)

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    logger.info(f"Total optimizer steps: {total_steps} | warmup steps: {warmup_steps}")

    global_step = 0
    best_metric = -1.0
    if train_cfg.resume_from_checkpoint:
        global_step, best_metric = load_checkpoint(
            train_cfg.resume_from_checkpoint, model, optimizer, scheduler, logger
        )

    # ---- Training loop ----
    model.train()
    running_loss, running_acc = 0.0, 0.0
    t0 = time.time()
    step_in_accum = 0

    data_iter = iter(train_loader)
    while global_step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        query_input_ids = batch["query_input_ids"].to(device)
        query_attention_mask = batch["query_attention_mask"].to(device)
        doc_input_ids = batch["doc_input_ids"].to(device)
        doc_attention_mask = batch["doc_attention_mask"].to(device)
        positive_indices = batch["positive_indices"].to(device)

        query_emb = model(query_input_ids, query_attention_mask)  # fp32, autograd-tracked

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            doc_emb = encode_documents(doc_model, doc_input_ids, doc_attention_mask)
        doc_emb = doc_emb.float()  # no grad here anyway (frozen), cast up for stable loss compute

        loss, acc = symmetric_infonce_loss(query_emb, doc_emb, positive_indices, train_cfg.temperature)
        (loss / train_cfg.gradient_accumulation_steps).backward()

        running_loss += loss.item()
        running_acc += acc
        step_in_accum += 1

        if step_in_accum == train_cfg.gradient_accumulation_steps:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), train_cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step_in_accum = 0
            global_step += 1

            if global_step % train_cfg.logging_steps == 0:
                elapsed = time.time() - t0
                avg_loss = running_loss / (train_cfg.logging_steps * train_cfg.gradient_accumulation_steps)
                avg_acc = running_acc / (train_cfg.logging_steps * train_cfg.gradient_accumulation_steps)
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"step={global_step}/{total_steps} loss={avg_loss:.4f} acc@1={avg_acc:.4f} "
                    f"lr={lr:.2e} grad_norm={grad_norm:.3f} sec/step={elapsed / train_cfg.logging_steps:.2f}"
                )
                running_loss, running_acc = 0.0, 0.0
                t0 = time.time()

            if global_step % train_cfg.eval_steps == 0:
                metric = evaluate(model, doc_model, val_loader, device, train_cfg.temperature, logger)
                if metric > best_metric:
                    best_metric = metric
                    logger.info(f"New best acc@1={best_metric:.4f} at step {global_step}")

            if global_step % train_cfg.save_steps == 0:
                save_checkpoint(
                    train_cfg.output_dir, global_step, model, optimizer, scheduler,
                    best_metric, train_cfg.save_total_limit, logger
                )

    # Final checkpoint
    save_checkpoint(train_cfg.output_dir, global_step, model, optimizer, scheduler,
                     best_metric, train_cfg.save_total_limit, logger)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
