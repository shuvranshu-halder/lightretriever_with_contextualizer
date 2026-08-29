#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
Query-side Contextualizer for LightRetriever.

Pipeline:
    query tokens
        -> frozen per-token lookup (nn.Embedding, NOT nn.EmbeddingBag, since
           we need per-token vectors here rather than a pre-summed bag)
        -> + learned absolute positional embeddings
        -> N-layer bidirectional (non-causal) Transformer encoder
        -> masked mean pooling
        -> (optional) L2 normalize
        => final query embedding
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class TransformerEncoderLayer(nn.Module):
    """Pre-norm bidirectional Transformer encoder block."""

    def __init__(self, hidden_size: int, num_heads: int, ffn_multiplier: int, dropout: float, activation: str):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_size)
        ffn_dim = ffn_multiplier * hidden_size
        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # key_padding_mask: True at PAD positions (to be ignored), shape [B, L]
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop1(attn_out)

        h = self.norm2(x)
        x = x + self.drop2(self.ffn(h))
        return x


class QueryContextualizer(nn.Module):
    def __init__(self, cfg: ModelConfig, emb_bag_weight: torch.Tensor, pad_token_id: int):
        super().__init__()
        vocab_size, hidden_size = emb_bag_weight.shape
        assert hidden_size % cfg.num_heads == 0, (
            f"hidden_size={hidden_size} must be divisible by num_heads={cfg.num_heads}"
        )
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id

        # ---- Frozen lookup table -------------------------------------------------
        # Loaded directly from the EmbeddingBag weight cached in cache_emb_bag.ipynb.
        # We use a plain nn.Embedding here (per-token vectors), NOT nn.EmbeddingBag,
        # because the contextualizer needs individual token embeddings, not a
        # pre-summed/averaged bag.
        self.token_embedding = nn.Embedding.from_pretrained(
            emb_bag_weight.clone(), freeze=True, padding_idx=pad_token_id
        )

        # ---- Trainable positional embedding ---------------------------------------
        self.position_embedding = nn.Embedding(cfg.max_query_len, hidden_size)
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.emb_layernorm = nn.LayerNorm(hidden_size)

        # ---- Trainable contextualizer ----------------------------------------------
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size=hidden_size,
                num_heads=cfg.num_heads,
                ffn_multiplier=cfg.ffn_multiplier,
                dropout=cfg.dropout,
                activation=cfg.activation,
            )
            for _ in range(cfg.num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)

        self._init_trainable_weights()

    def _init_trainable_weights(self):
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def trainable_parameters(self):
        """Convenience: only params that require grad (excludes frozen lookup table)."""
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] token ids (query tokens only, no prompt needed here -
                       the "prompt" semantics is already baked into the EmbeddingBag
                       weights themselves, since each row of the table was built as
                       eos-pooled([bos]+prompt+[vocab_token]+[eos]).)
            attention_mask: [B, L] bool/int, 1 for real tokens, 0 for padding.
        Returns:
            query_embedding: [B, hidden_size], L2-normalized.
        """
        B, L = input_ids.shape
        device = input_ids.device

        with torch.no_grad():
            tok_emb = self.token_embedding(input_ids)  # [B, L, H], frozen

        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        pos_emb = self.position_embedding(pos_ids)      # [B, L, H], trainable

        x = self.emb_layernorm(tok_emb + pos_emb)
        x = self.emb_dropout(x)

        key_padding_mask = ~attention_mask.bool()  # True = PAD (ignored by attention)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        x = self.final_norm(x)

        # Masked mean pooling
        mask_f = attention_mask.unsqueeze(-1).to(x.dtype)  # [B, L, 1]
        summed = (x * mask_f).sum(dim=1)
        counts = mask_f.sum(dim=1).clamp(min=1e-6)
        pooled = summed / counts

        return F.normalize(pooled, p=2, dim=-1)
