"""
GrokLLM — Minimal Decoder-Only Transformer

This is the complete neural network implementation.
Written 100% autonomously by Grok Build CLI in YOLO mode.

Architecture: GPT-style
- Learned token + position embeddings
- Stack of Transformer blocks with causal self-attention
- Pre-LayerNorm + residual connections
- No flash attention, no KV cache (model is tiny by design)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import GrokLLMConfig, DEFAULT_CONFIG


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: GrokLLMConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Combined QKV projection for efficiency
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask (lower triangular)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, time, channels (n_embd)

        # Calculate Q, K, V
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head: (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        # Apply causal mask
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of values
        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y


class MLP(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, config: GrokLLMConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """One Transformer block: attention + MLP with pre-LayerNorm and residuals."""

    def __init__(self, config: GrokLLMConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GrokLLM(nn.Module):
    """
    The full GrokLLM model.

    A miniature decoder-only Transformer language model.
    Designed to be trained from scratch on a CPU in minutes.
    """

    def __init__(self, config: Optional[GrokLLMConfig] = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG

        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.position_embedding = nn.Embedding(self.config.block_size, self.config.n_embd)
        self.dropout = nn.Dropout(self.config.dropout)

        self.blocks = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)])
        self.ln_f = nn.LayerNorm(self.config.n_embd)

        # Language modeling head (tied weights with token embedding is an option but we keep separate for clarity)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GrokLLM initialized with {n_params:,} parameters")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            idx: (B, T) token indices
            targets: (B, T) optional next-token targets for loss

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar or None
        """
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"

        # Embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long)
        pos_emb = self.position_embedding(pos)  # (T, n_embd)
        x = self.dropout(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        Args:
            idx: (B, T) conditioning tokens
            max_new_tokens: how many new tokens to generate
            temperature: softmax temperature (>0)
            top_k: if set, only sample from top-k logits
            top_p: if set, nucleus sampling (cumulative prob)
            eos_token_id: optional early stop

        Returns:
            (B, T + generated) token ids
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)  # last position

            # Top-k filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p is not None and 0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative prob > top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat((idx, next_token), dim=1)

            # Early stop on EOS if provided
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return idx

    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None,
                        step: int = 0, best_val_loss: float = float("inf"),
                        tokenizer_stoi: Optional[dict] = None,
                        tokenizer_itos: Optional[list] = None) -> None:
        """Save model + training state."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config.__dict__,
            "step": step,
            "best_val_loss": best_val_loss,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if tokenizer_stoi is not None:
            checkpoint["stoi"] = tokenizer_stoi
        if tokenizer_itos is not None:
            checkpoint["itos"] = tokenizer_itos

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> Tuple["GrokLLM", dict]:
        """Load model from checkpoint. Returns (model, extra_state)."""
        checkpoint = torch.load(path, map_location=device)

        cfg_dict = checkpoint.get("config", {})
        config = GrokLLMConfig(**{k: v for k, v in cfg_dict.items() if hasattr(GrokLLMConfig, k)})

        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        extra = {
            "step": checkpoint.get("step", 0),
            "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
            "stoi": checkpoint.get("stoi"),
            "itos": checkpoint.get("itos"),
            "optimizer_state_dict": checkpoint.get("optimizer_state_dict"),
        }
        return model, extra
