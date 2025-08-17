#!/usr/bin/env python
"""
Minimalist Gemma‑style Decoder LM trainer in pure PyTorch (no Transformers)

What this is:
- A compact, readable, single‑file trainer for a decoder‑only Transformer
- Gemma‑inspired blocks: RMSNorm, Rotary Positional Embedding (RoPE), SwiGLU MLP
- Byte tokenizer by default (no external deps); optional SentencePiece (.model) support
- Mixed precision, gradient accumulation, cosine LR, checkpointing, simple text gen

What this is NOT:
- A drop‑in reproduction of Google Gemma weights/architecture
- Highly optimized kernels (no FlashAttention, no fused ops)

Quickstart
----------
# 1) Prepare a plain text file `data.txt` (each line = a sample) OR your own loader
# 2) (Optional) Train/obtain a SentencePiece model `tok.model` and pass --spm tok.model
# 3) Train a tiny model on CPU/GPU:
#    python gemma_min_trainer.py \
#      --data data.txt --out out_tiny --vocab_size 256 \
#      --n_layer 4 --n_head 6 --d_model 384 --d_ff 2048 \
#      --seq_len 512 --batch_size 16 --grad_accum 2 --epochs 2 --lr 3e-4 --bf16
# 4) Generate:
#    python gemma_min_trainer.py --out out_tiny --generate "Merhaba" --max_new_tokens 100 --top_k 50

Notes
-----
- If you pass --spm path/to/model, SentencePiece is used (pip install sentencepiece). Otherwise a byte tokenizer is used.
- This file aims for clarity over speed; scale up gradually and consider replacing attention with efficient kernels for large models.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------
# Tokenizers
# ---------------------
class ByteTokenizer:
    """Simple reversible byte tokenizer with 256 base tokens + specials."""
    def __init__(self):
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.offset = 4
        self.vocab_size = 256 + self.offset

    def encode(self, s: str, add_bos=False, add_eos=False) -> List[int]:
        ids = [self.bos_id] if add_bos else []
        ids += [b + self.offset for b in s.encode("utf-8", errors="replace")]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        bytes_list = []
        for t in ids:
            if t >= self.offset:
                bytes_list.append(max(0, min(255, t - self.offset)))
        return bytes(bytes_list).decode("utf-8", errors="replace")

try:
    import sentencepiece as spm  # optional
except Exception:
    spm = None

class SentencePieceTokenizer:
    def __init__(self, model_path: str):
        assert spm is not None, "Install sentencepiece or omit --spm."
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else 0
        self.bos_id = self.sp.bos_id() if self.sp.bos_id() >= 0 else 1
        self.eos_id = self.sp.eos_id() if self.sp.eos_id() >= 0 else 2
        self.unk_id = self.sp.unk_id() if self.sp.unk_id() >= 0 else 3
        self.vocab_size = self.sp.vocab_size()

    def encode(self, s: str, add_bos=False, add_eos=False) -> List[int]:
        ids = self.sp.encode(s, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        return self.sp.decode(ids)

# ---------------------
# Data
# ---------------------
class LineByLineTextDataset(Dataset):
    def __init__(self, path: str, tokenizer, seq_len: int):
        self.lines = [l.rstrip("\n") for l in open(path, "r", encoding="utf-8") if len(l.strip()) > 0]
        self.tok = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]
        ids = self.tok.encode(text, add_bos=True, add_eos=True)
        # Crop / pad to seq_len
        if len(ids) < self.seq_len + 1:
            ids = ids + [self.tok.pad_id] * (self.seq_len + 1 - len(ids))
        else:
            ids = ids[: self.seq_len + 1]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        attn_mask = (x != self.tok.pad_id).to(torch.bool)
        return x, y, attn_mask

# ---------------------
# Model bits (Gemma‑ish)
# ---------------------
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # x: (B,T,C)
        norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * norm_x

class RoPE:
    def __init__(self, dim: int, base: float = 10000.0):
        self.dim = dim
        self.base = base

    def get_cos_sin(self, seq_len: int, device, dtype):
        # Build rotary frequencies for half-dim pairs
        half = self.dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("t,f->tf", t, inv_freq)  # (T, half)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        # shape (T, dim) by interleaving
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        return cos, sin

    @staticmethod
    def apply_rotary(x, cos, sin):
        # x: (B, n_head, T, head_dim)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rot = torch.stack([-x2, x1], dim=-1).reshape_as(x)
        return x * cos + x_rot * sin

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, rope_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RoPE(rope_dim or self.head_dim)

        # causal mask cached lazily
        self.register_buffer("_mask", None, persistent=False)

    def _get_causal_mask(self, T, device):
        if self._mask is None or self._mask.size(0) < T:
            mask = torch.full((T, T), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self._mask = mask.to(device)
        return self._mask[:T, :T]

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B,T,3C)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B,h,T,hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # RoPE
        cos, sin = self.rope.get_cos_sin(T, x.device, x.dtype)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,hd)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = RoPE.apply_rotary(q, cos, sin)
        k = RoPE.apply_rotary(k, cos, sin)

        # scaled dot‑product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,h,T,T)
        att = att + self._get_causal_mask(T, x.device)
        if attn_mask is not None:
            # attn_mask: (B,T) True for valid tokens; broadcast to (B,1,1,T)
            mask = attn_mask[:, None, None, :].to(att.dtype)
            att = att + (mask - 1.0) * 1e10  # large negative where padded
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v  # (B,h,T,hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, attn_dropout=0.0, resid_dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_head, dropout=attn_dropout)
        self.drop1 = nn.Dropout(resid_dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_ff, dropout=resid_dropout)
        self.drop2 = nn.Dropout(resid_dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.drop1(self.attn(self.norm1(x), attn_mask=attn_mask))
        x = x + self.drop2(self.mlp(self.norm2(x)))
        return x

class TinyGemmaLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int=512, n_layer: int=8, n_head: int=8, d_ff: int=2048, resid_dropout=0.0, attn_dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_head, d_ff, attn_dropout=attn_dropout, resid_dropout=resid_dropout)
            for _ in range(n_layer)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, attn_mask=None):
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

# helper to build model from a config-like object
def build_model_from_cfg(cfg_like, vocab_size: int):
    d_model = getattr(cfg_like, 'd_model', 512)
    n_layer = getattr(cfg_like, 'n_layer', 8)
    n_head  = getattr(cfg_like, 'n_head', 8)
    d_ff    = getattr(cfg_like, 'd_ff', 2048)
    return TinyGemmaLM(
        vocab_size=vocab_size,
        d_model=d_model, n_layer=n_layer, n_head=n_head, d_ff=d_ff
    )

# ---------------------
# Training utilities
# ---------------------
@dataclass
class Config:
    data: str = "data.txt"
    out: str = "out"
    spm: Optional[str] = None
    vocab_size: int = 256
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    d_ff: int = 2048
    seq_len: int = 512
    batch_size: int = 8
    grad_accum: int = 1
    epochs: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    cosine_min_lr_ratio: float = 0.1
    seed: int = 42
    bf16: bool = False
    fp16: bool = False
    # generation-related (were missing before)
    generate: Optional[str] = None
    max_new_tokens: int = 100
    top_k: Optional[int] = None
    device: str = "auto"  # Will be set to detected device


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def cosine_scheduler(optimizer, total_steps, warmup_steps, min_lr_ratio):
    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, step, cfg: Config, tok_meta: dict):
    outdir = Path(cfg.out)
    outdir.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "step": step,
        "config": asdict(cfg),
        "tokenizer": tok_meta,
    }
    torch.save(state, outdir / "ckpt.pt")


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None):
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])  # strict
    if optimizer is not None and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    return state

# ---------------------
# Generation (greedy/top‑k)
# ---------------------
@torch.no_grad()
def generate(model: TinyGemmaLM, tok, prompt: str, max_new_tokens=100, temperature=1.0, top_k: Optional[int]=None, device="cpu"):
    model.eval()
    idx = torch.tensor([tok.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
    attn_mask = (idx != tok.pad_id)
    for _ in range(max_new_tokens):
        logits = model(idx, attn_mask=attn_mask)[:, -1, :] / max(1e-6, temperature)
        if top_k is not None:
            v, _ = torch.topk(logits, k=top_k)
            logits = torch.where(logits < v[:, [-1]], torch.full_like(logits, -1e10), logits)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
        attn_mask = (idx != tok.pad_id)
        if next_id.item() == tok.eos_id:
            break
    return tok.decode(idx[0].tolist())

# ---------------------
# Main script
# ---------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data.txt")
    p.add_argument("--out", type=str, default="out")
    p.add_argument("--spm", type=str, default=None, help="Path to SentencePiece model (.model)")
    p.add_argument("--vocab_size", type=int, default=256, help="Used only for byte tokenizer (fixed=260 incl specials)")
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--d_ff", type=int, default=2048)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--cosine_min_lr_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--generate", type=str, default=None, help="Generate instead of train (uses latest ckpt)")
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--top_k", type=int, default=None)
    args = p.parse_args()
    return Config(**vars(args))


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    cfg = parse_args()
    # Set device to detected device if auto
    if cfg.device == "auto":
        cfg.device = detect_device()
    print(f"Using device: {cfg.device}")
    set_seed(cfg.seed)

    # Tokenizer
    if cfg.spm:
        tok = SentencePieceTokenizer(cfg.spm)
    else:
        tok = ByteTokenizer()
    vocab_size = tok.vocab_size

    # Model
    model = TinyGemmaLM(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        d_ff=cfg.d_ff,
        resid_dropout=0.0,
        attn_dropout=0.0,
    ).to(cfg.device)

    outdir = Path(cfg.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if cfg.generate:
        ckpt_path = Path(cfg.out) / "ckpt.pt"
        assert ckpt_path.exists(), f"No checkpoint at {ckpt_path}"
        state = torch.load(ckpt_path, map_location="cpu")
        saved_cfg_dict = state.get("config", {})

        # Recreate a lightweight object with saved hyperparams
        class _C: pass
        saved = _C()
        for k, v in saved_cfg_dict.items():
            setattr(saved, k, v)

        # Tokenizer: prefer SentencePiece if checkpoint says so and --spm was provided
        tok_type = (state.get("tokenizer", {}) or {}).get("type", "ByteTokenizer")
        if tok_type == "SentencePieceTokenizer":
            if cfg.spm is None:
                print("[warn] Checkpoint used SentencePiece but --spm not provided. Falling back to byte tokenizer; decoding may differ.")
                tok = ByteTokenizer()
            else:
                tok = SentencePieceTokenizer(cfg.spm)
        else:
            tok = ByteTokenizer()

        # Build model to match checkpointed architecture, then load weights
        model = build_model_from_cfg(saved, vocab_size=tok.vocab_size).to(cfg.device)
        model.load_state_dict(state["model"])  # strict load OK now

        text = generate(
            model, tok, cfg.generate,
            max_new_tokens=cfg.max_new_tokens, top_k=cfg.top_k, device=cfg.device
        )
        print(text)
        return

    # Data
    ds = LineByLineTextDataset(cfg.data, tok, cfg.seq_len)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * (len(dl) // max(1, cfg.grad_accum))
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    scheduler = cosine_scheduler(optimizer, total_steps, warmup_steps, cfg.cosine_min_lr_ratio)

    # MPS-compatible autocast and scaler
    if cfg.device == "cuda":
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)
        use_autocast = cfg.fp16 or cfg.bf16
        autocast_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    elif cfg.device == "mps":
        scaler = None
        use_autocast = cfg.bf16  # MPS supports bfloat16
        autocast_dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    else:
        scaler = None
        use_autocast = False
        autocast_dtype = torch.float32

    # Training loop
    model.train()
    global_step = 0
    best_loss = float("inf")
    t0 = time.time()
    for epoch in range(cfg.epochs):
        running = 0.0
        for step, batch in enumerate(dl):
            x, y, attn_mask = [t.to(cfg.device) for t in batch]

            # MPS-compatible autocast
            if cfg.device == "cuda":
                autocast_context = torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype)
            else:
                autocast_context = torch.autocast(device_type=cfg.device, enabled=use_autocast, dtype=autocast_dtype)
            
            with autocast_context:
                logits = model(x, attn_mask=attn_mask)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=tok.pad_id)
                loss = loss / cfg.grad_accum

            # MPS-compatible backward pass
            if cfg.device == "cuda" and cfg.fp16 and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % cfg.grad_accum == 0:
                if cfg.device == "cuda" and cfg.fp16 and scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if cfg.device == "cuda" and cfg.fp16 and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            running += loss.item() * cfg.grad_accum

            if global_step % 50 == 0:
                elapsed = time.time() - t0
                lr = scheduler.get_last_lr()[0]
                print(f"epoch {epoch+1} step {global_step}/{total_steps} | loss {running/50:.4f} | lr {lr:.2e} | {elapsed:.1f}s")
                running = 0.0

            if global_step > 0 and global_step % 500 == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, cfg, tok_meta={"type": tok.__class__.__name__})

        # end epoch
        # quick val on a random minibatch (optional)
        model.eval()
        with torch.no_grad():
            x, y, attn_mask = [t.to(cfg.device) for t in next(iter(dl))]
            logits = model(x, attn_mask=attn_mask)
            val_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=tok.pad_id).item()
        model.train()
        print(f"Epoch {epoch+1} done. Val loss: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, global_step, cfg, tok_meta={"type": tok.__class__.__name__})

    # final save
    save_checkpoint(model, optimizer, scheduler, global_step, cfg, tok_meta={"type": tok.__class__.__name__})
    print("Training complete.")

if __name__ == "__main__":
    main()
