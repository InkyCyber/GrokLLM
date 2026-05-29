"""
GrokLLM Character-Level Tokenizer

Written entirely by Grok Build CLI (no human code).
A deliberately simple tokenizer so the entire pipeline remains educational.
"""

from typing import List, Dict, Tuple
import os


class CharTokenizer:
    """
    Character-level tokenizer.

    Builds vocabulary from raw text. Supports encode/decode.
    Stores stoi (char -> int) and itos (int -> char) mappings.
    """

    def __init__(self, text: str = None):
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []
        self.vocab_size: int = 0

        if text is not None:
            self.build_vocab(text)

    def build_vocab(self, text: str) -> None:
        """Build vocabulary from the given text corpus."""
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = chars
        self.vocab_size = len(chars)

    def encode(self, s: str) -> List[int]:
        """Encode a string into a list of token ids."""
        if not self.stoi:
            raise ValueError("Tokenizer vocabulary has not been built or loaded.")
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        """Decode a list of token ids back into a string."""
        if not self.itos:
            raise ValueError("Tokenizer vocabulary has not been built or loaded.")
        return "".join(self.itos[i] for i in ids if 0 <= i < len(self.itos))

    def save(self, path: str) -> None:
        """Save tokenizer state to a file."""
        import json
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "stoi": self.stoi,
                "itos": self.itos,
                "vocab_size": self.vocab_size
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """Load tokenizer from a saved file."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.stoi = data["stoi"]
        tok.itos = data["itos"]
        tok.vocab_size = data["vocab_size"]
        return tok

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"


def get_tokenizer_for_text(text: str) -> CharTokenizer:
    """Convenience: create and build a tokenizer from raw text."""
    tok = CharTokenizer()
    tok.build_vocab(text)
    return tok
