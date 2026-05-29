"""
High-level generation interface for GrokLLM.

Provides a clean generate_text() function used by CLI and web server.
Completely written by Grok Build CLI — no human code.
"""

import os
import time
from typing import Optional, Dict, Any

import torch

from .config import GrokLLMConfig, DEFAULT_CONFIG
from .model import GrokLLM
from .tokenizer import CharTokenizer


class GrokGenerator:
    """Convenience wrapper around a loaded model + tokenizer for text generation."""

    def __init__(self, model: GrokLLM, tokenizer: CharTokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: Optional[str] = None) -> "GrokGenerator":
        """Load generator directly from a checkpoint file."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model, extra = GrokLLM.load_checkpoint(checkpoint_path, device=device)

        if extra.get("stoi") and extra.get("itos"):
            tok = CharTokenizer()
            tok.stoi = extra["stoi"]
            tok.itos = extra["itos"]
            tok.vocab_size = len(extra["itos"])
        else:
            raise ValueError("Checkpoint is missing tokenizer vocabulary (stoi/itos).")

        return cls(model, tok, device=device)

    def generate(
        self,
        prompt: str = "",
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate text from a prompt.

        Returns a rich dictionary with text, timing, and metadata.
        """
        if seed is not None:
            torch.manual_seed(seed)

        start_time = time.time()

        # Encode prompt
        if prompt:
            idx = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device)
        else:
            # Start from a newline or random token if no prompt
            idx = torch.zeros((1, 1), dtype=torch.long, device=self.device)

        # Generate
        with torch.no_grad():
            out = self.model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        # Decode
        generated_ids = out[0].tolist()
        text = self.tokenizer.decode(generated_ids)

        elapsed = time.time() - start_time
        tokens_generated = len(generated_ids) - idx.size(1)
        tokens_per_sec = tokens_generated / max(elapsed, 1e-6)

        return {
            "text": text,
            "prompt": prompt,
            "generated_continuation": text[len(prompt):] if prompt else text,
            "tokens_generated": tokens_generated,
            "elapsed_seconds": round(elapsed, 3),
            "tokens_per_second": round(tokens_per_sec, 1),
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "seed": seed,
            },
            "model_info": {
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "block_size": self.model.config.block_size,
                "n_layer": self.model.config.n_layer,
            }
        }


def generate_text(
    prompt: str = "",
    checkpoint_path: Optional[str] = None,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = 0.9,
    device: Optional[str] = None,
) -> str:
    """
    One-shot generation function. Loads model if needed.

    This is the simplest entry point for scripts.
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(DEFAULT_CONFIG.checkpoint_dir, DEFAULT_CONFIG.checkpoint_name)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            "Run `python train.py` first to create one."
        )

    gen = GrokGenerator.from_checkpoint(checkpoint_path, device=device)
    result = gen.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    return result["text"]


def main():
    """CLI entry for generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate text with GrokLLM")
    parser.add_argument("--prompt", "-p", type=str, default="To be or not to be", help="Prompt text")
    parser.add_argument("--max-tokens", "-m", type=int, default=200)
    parser.add_argument("--temperature", "-t", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = os.path.join(DEFAULT_CONFIG.checkpoint_dir, DEFAULT_CONFIG.checkpoint_name)

    print("=" * 70)
    print("GrokLLM Generator — Built entirely by Grok Build CLI (YOLO mode)")
    print("=" * 70)
    print(f"Prompt: {args.prompt!r}")
    print(f"Params: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, max={args.max_tokens}")
    print("-" * 70)

    gen = GrokGenerator.from_checkpoint(args.checkpoint, device=args.device)
    result = gen.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )

    print(result["text"])
    print("-" * 70)
    print(f"Generated {result['tokens_generated']} tokens in {result['elapsed_seconds']}s "
          f"({result['tokens_per_second']} tok/s)")
    print("=" * 70)


if __name__ == "__main__":
    main()
