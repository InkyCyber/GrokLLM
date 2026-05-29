#!/usr/bin/env python3
"""
GrokLLM Training Script

Trains a tiny decoder-only Transformer on Tiny Shakespeare (or your own .txt).

This script, like everything else in the project, was written 100% autonomously
by the Grok Build CLI during the original YOLO-mode challenge.

Typical runtime on a modern laptop CPU: 45–120 seconds for a usable model.
"""

import os
import time
import math
from contextlib import nullcontext

import torch

from grokllm.config import GrokLLMConfig, DEFAULT_CONFIG
from grokllm.model import GrokLLM
from grokllm.tokenizer import CharTokenizer, get_tokenizer_for_text
from grokllm.data import load_text_data, prepare_dataset, get_batch, estimate_loss


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(config: GrokLLMConfig = None):
    cfg = config or DEFAULT_CONFIG
    device = get_device()
    print(f"Using device: {device}")
    print(f"Model config: layers={cfg.n_layer}, heads={cfg.n_head}, embd={cfg.n_embd}, block={cfg.block_size}")

    # 1. Data
    print("\n[1/5] Loading data...")
    text = load_text_data(cfg)
    print(f"  Loaded {len(text):,} characters")

    # 2. Tokenizer
    print("[2/5] Building tokenizer...")
    tokenizer = get_tokenizer_for_text(text)
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    cfg.vocab_size = tokenizer.vocab_size  # important: update config

    # 3. Dataset splits
    train_data, val_data = prepare_dataset(text, tokenizer, cfg)
    print(f"  Train tokens: {len(train_data):,}")
    print(f"  Val tokens:   {len(val_data):,}")

    # 4. Model
    print("\n[3/5] Initializing model...")
    model = GrokLLM(cfg).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer (AdamW with weight decay on non-bias/ln params)
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=cfg.learning_rate, betas=(0.9, 0.99))

    # 5. Training loop
    print("\n[4/5] Training...")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)

    model.train()
    best_val = float("inf")
    start_time = time.time()

    # Simple linear warmup + cosine decay
    def get_lr(step):
        if step < cfg.warmup_iters:
            return cfg.learning_rate * (step + 1) / max(cfg.warmup_iters, 1)
        progress = (step - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
        return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    ctx = nullcontext() if device == "cpu" else torch.autocast(device_type=device, dtype=torch.float16)

    for step in range(cfg.max_iters):
        # Learning rate schedule
        lr = get_lr(step)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Batch
        x, y = get_batch(train_data, cfg.batch_size, cfg.block_size, device)

        with ctx:
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Logging & evaluation
        if step % 50 == 0 or step == cfg.max_iters - 1:
            elapsed = time.time() - start_time
            print(f"  step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.1f}s", end="")

            if step % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
                train_loss, val_loss = estimate_loss(model, train_data, val_data, cfg, device, cfg.eval_iters)
                print(f"  | train {train_loss:.4f} val {val_loss:.4f}", end="")

                if val_loss < best_val:
                    best_val = val_loss
                    model.save_checkpoint(
                        checkpoint_path,
                        optimizer=optimizer,
                        step=step,
                        best_val_loss=best_val,
                        tokenizer_stoi=tokenizer.stoi,
                        tokenizer_itos=tokenizer.itos,
                    )
                    print("  [saved best]", end="")
            print()

    total_time = time.time() - start_time
    print(f"\n[5/5] Training complete in {total_time:.1f}s")
    print(f"Best validation loss: {best_val:.4f}")
    print(f"Checkpoint: {checkpoint_path}")

    # Final quick generation test
    print("\n--- Quick sample from the newly trained model ---")
    model.eval()
    with torch.no_grad():
        # start with a capital letter context
        start = torch.tensor([[tokenizer.encode("The ")[0]]], device=device)
        out = model.generate(start, max_new_tokens=120, temperature=0.8, top_k=40)
        sample = tokenizer.decode(out[0].tolist())
        print(sample)
        print("---")

    print("\nDone. You can now run the web UI:")
    print("  python -m grokllm.web")
    print("  or: python app.py")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train GrokLLM (tiny Shakespeare Transformer)")
    parser.add_argument("--max-iters", type=int, default=None, help="Override training steps")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-layer", type=int, default=None, help="Number of layers (for experimentation)")
    parser.add_argument("--n-embd", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    args = parser.parse_args()

    cfg = GrokLLMConfig()
    if args.max_iters: cfg.max_iters = args.max_iters
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.n_layer: cfg.n_layer = args.n_layer
    if args.n_embd: cfg.n_embd = args.n_embd
    if args.n_head: cfg.n_head = args.n_head
    if args.block_size: cfg.block_size = args.block_size

    train(cfg)
