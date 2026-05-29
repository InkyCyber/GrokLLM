"""
GrokLLM Configuration

This entire module (and all of GrokLLM) was written autonomously
by the Grok Build CLI in YOLO mode. No human wrote this code.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GrokLLMConfig:
    """Configuration for the tiny GrokLLM Transformer."""

    # Model architecture
    block_size: int = 128          # max context length
    n_layer: int = 4               # number of Transformer blocks
    n_head: int = 4                # number of attention heads
    n_embd: int = 128              # embedding dimension
    dropout: float = 0.1

    # Vocabulary (set dynamically from data or loaded from checkpoint)
    vocab_size: int = 65           # will be overridden by tokenizer

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_iters: int = 4000
    warmup_iters: int = 200
    eval_interval: int = 200
    eval_iters: int = 50
    batch_size: int = 32

    # Generation defaults
    temperature: float = 0.8
    top_k: Optional[int] = 40
    top_p: Optional[float] = 0.9
    max_new_tokens: int = 200

    # Data
    data_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_path: str = "data/tiny_shakespeare.txt"

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "grokllm_tiny.pt"

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"


# Default config instance used everywhere
DEFAULT_CONFIG = GrokLLMConfig()
