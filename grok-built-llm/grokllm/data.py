"""
Data loading utilities for GrokLLM.

Handles downloading Tiny Shakespeare and creating training batches.
100% written by Grok Build CLI.
"""

import os
import requests
import torch
from typing import Tuple

from .config import GrokLLMConfig
from .tokenizer import CharTokenizer


def download_if_missing(url: str, path: str) -> str:
    """Download file from URL if it doesn't exist locally. Returns the file path."""
    if os.path.exists(path):
        return path

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"Downloading dataset from {url} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    print(f"Saved to {path} ({len(resp.text):,} chars)")
    return path


def load_text_data(config: GrokLLMConfig) -> str:
    """Load the full training text content (downloads if needed)."""
    path = download_if_missing(config.data_url, config.data_path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (x, y) sequences."""
    n = len(data) - block_size - 1
    if n <= 0:
        raise ValueError(f"Dataset too small for block_size={block_size}. Need at least {block_size + 2} tokens.")
    ix = torch.randint(0, max(n, 1), (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y


def prepare_dataset(
    text: str,
    tokenizer: CharTokenizer,
    config: GrokLLMConfig,
    val_split: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize text and split into train/val tensors.
    Returns (train_data, val_data) as long tensors.
    """
    tokens = tokenizer.encode(text)
    data = torch.tensor(tokens, dtype=torch.long)

    n = int(len(data) * (1 - val_split))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data


def estimate_loss(
    model: torch.nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    config: GrokLLMConfig,
    device: str,
    eval_iters: int = 50
) -> Tuple[float, float]:
    """Estimate train and val loss over a few batches."""
    model.eval()
    losses = {"train": torch.zeros(eval_iters), "val": torch.zeros(eval_iters)}

    for split, data in [("train", train_data), ("val", val_data)]:
        for k in range(eval_iters):
            x, y = get_batch(data, config.batch_size, config.block_size, device)
            with torch.no_grad():
                _, loss = model(x, y)
            losses[split][k] = loss.item()

    model.train()
    return losses["train"].mean().item(), losses["val"].mean().item()
