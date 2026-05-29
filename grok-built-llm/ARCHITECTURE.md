# Architecture

This document describes the technical design of GrokLLM as implemented autonomously by the Grok Build CLI.

## Design Philosophy

The agent was given an ambitious but time-bounded task. The architecture prioritizes:

- **Educational clarity** over SOTA performance
- **Single-file understandability** for the core model
- **Zero-config first run** experience
- **Complete self-containment** (train + serve + UI from one checkout)
- **Strong visual and textual attribution** to the autonomous build process

## Model Architecture

GrokLLM implements a **decoder-only Transformer** (GPT-style) at miniature scale.

### Configuration (default)

```python
block_size: 128          # context length
n_layer: 4               # Transformer blocks
n_head: 4                # attention heads
n_embd: 128              # embedding dimension
dropout: 0.1
vocab_size: 65           # character level (Tiny Shakespeare)
```

**Approximate parameter count: ~800,000**

### Block Structure (per layer)

1. LayerNorm
2. Causal Multi-Head Self-Attention (with `nn.MultiheadAttention` + causal mask via `attn_mask`)
3. Residual connection + dropout
4. LayerNorm
5. Position-wise MLP (Linear → GELU → Linear)
6. Residual connection + dropout

### Positional Encoding

Learned positional embeddings (not sinusoidal). Simple and sufficient at this scale.

### Tokenization

Pure character-level tokenizer.

- Vocabulary is built dynamically from the training text
- Default run uses Tiny Shakespeare (~1M characters → 65 unique tokens)
- Tokenizer is deliberately trivial: `char_to_idx`, `idx_to_char`, no BPE, no SentencePiece

This choice was made so that:
- Training finishes in < 2 minutes on CPU
- The generated text is still recognizably "language modeling"
- Anyone can inspect every token

## Training

- Optimizer: AdamW (weight decay 0.01 on non-bias/LayerNorm params)
- Learning rate: 3e-4 with cosine decay + warmup (simple schedule)
- Loss: Cross-entropy over next character
- Data: Random contiguous chunks of size `block_size + 1`
- No gradient accumulation needed at this scale
- Checkpointing: saves full model state + optimizer + step + best val loss

The training script (`train.py`) is intentionally short (~150 lines) and heavily commented.

## Generation / Sampling

Implemented in `grokllm/generate.py`:

- Greedy (temperature = 0)
- Temperature scaling
- Top-k filtering
- Top-p (nucleus) filtering
- Repetition penalty (light)
- KV-cache is **not** implemented (model is tiny; not worth the complexity for this demo)

Generation returns both the text and token-level metadata.

## Web Architecture

### Backend (`app.py`)

- FastAPI
- Single `/generate` POST endpoint
- Request validation via Pydantic
- Model loaded once at startup (or lazy-loaded on first request)
- CORS enabled for local development
- Graceful handling when no checkpoint exists (returns friendly error + training hint)

### Frontend

A **single self-contained HTML file** (`web/index.html`):

- Tailwind CSS 3.4 via CDN (no build step)
- Vanilla JavaScript only
- No frameworks
- Responsive
- Dark theme with strong Grok branding accents
- Every major UI surface contains visible attribution text

The frontend deliberately avoids bundlers so that the entire UI can be read and understood in one file — consistent with the educational ethos.

### API Surface

See [API.md](API.md) for the full OpenAPI-style reference.

Primary endpoint:

```
POST /api/generate
{
  "prompt": "To be or not",
  "max_new_tokens": 200,
  "temperature": 0.8,
  "top_k": 40,
  "top_p": 0.9
}
```

## Data

- `data/tiny_shakespeare.txt` is downloaded on first `train.py` run from Karpathy's public gist
- The download is deterministic and cached
- Users can replace it with any `.txt` file (the tokenizer is rebuilt from whatever text is present)

## Checkpoints

Checkpoints are PyTorch `.pt` files containing:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "config": {...},
    "step": int,
    "best_val_loss": float,
    "vocab": {...},
    "itos": list,
    "stoi": dict
}
```

This makes checkpoints fully portable between training runs and the web server.

## Why Not Bigger?

Several deliberate constraints:

1. **CPU trainability in < 2 minutes** is a core user experience goal. A 7M+ param model would frustrate first-time users.
2. **Transparency**: At 800K parameters, an interested reader can actually load the weights in a notebook and inspect them.
3. **Agentic demonstration**: The point of this repo is not "we trained a good LLM". The point is "an agent built a complete, working, documented LLM system with zero human code."

A larger model would have required more training time, more complex distributed code, and would have diluted the core demonstration.

## Future Directions (Written by the Agent)

The agent noted several obvious extensions it could have implemented if given more time or a follow-up "fix" command:

- BPE tokenizer (but would have made the project less scrutable)
- Flash attention / better kernels (overkill)
- Real streaming generation via SSE (nice-to-have)
- LoRA fine-tuning UI
- Export to GGUF / ONNX
- Multi-GPU / MPS training path

None were implemented because the primary success criteria was **a complete, impressive, self-documenting system delivered autonomously**, not feature completeness.

## Testing & Verification

The agent performed the following validation steps during the build:

1. `python train.py` completes without error and produces a checkpoint with decreasing loss
2. `python -m grokllm.generate` produces coherent text
3. FastAPI server starts and `/docs` renders
4. Web UI loads and successfully calls the generate endpoint
5. All major attribution strings appear in the rendered HTML

No external test framework was added (to keep surface area minimal), but the code is structured for easy addition of pytest later.

---

**This architecture document was written 100% autonomously by Grok Build CLI during the original challenge session.**
