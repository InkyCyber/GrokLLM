"""
GrokLLM

A complete, minimal LLM + web interface built 100% autonomously
by the Grok Build CLI (Grok 4.3 by xAI) in YOLO / always-approve mode.

No human wrote any code in this repository except the original challenge prompt.
"""

__version__ = "0.1.0"
__author__ = "Grok Build CLI"
__build_mode__ = "YOLO / autonomous"

from .config import GrokLLMConfig, DEFAULT_CONFIG
from .model import GrokLLM
from .tokenizer import CharTokenizer
from .generate import GrokGenerator, generate_text

__all__ = [
    "GrokLLM",
    "GrokLLMConfig",
    "CharTokenizer",
    "GrokGenerator",
    "generate_text",
    "DEFAULT_CONFIG",
]
