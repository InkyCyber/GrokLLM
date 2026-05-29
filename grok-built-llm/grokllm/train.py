"""
Training entry point for installed script `grokllm-train`.

This file simply re-exports the root train.py logic.
All code 100% by Grok Build CLI.
"""

import sys
import os

# Allow running as module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from train import train as _train
from grokllm.config import GrokLLMConfig


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iters", type=int, default=None)
    args = parser.parse_args()

    cfg = GrokLLMConfig()
    if args.max_iters:
        cfg.max_iters = args.max_iters

    _train(cfg)


if __name__ == "__main__":
    main()
