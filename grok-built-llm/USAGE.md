# Usage Guide

Everything in this guide was written by Grok Build CLI.

## Installation

```bash
git clone https://github.com/johnperry/grok-built-llm.git
cd grok-built-llm
pip install -r requirements.txt
```

PyTorch is the only heavy dependency.

## 1. Train the Model

This is the actual step that creates a working neural network.

```bash
python train.py
```

What happens:
- Downloads Tiny Shakespeare (~1 MB)
- Builds a 65-token character vocabulary
- Trains a ~800K parameter Transformer for ~4000 steps
- Saves `checkpoints/grokllm_tiny.pt` when validation loss improves

**Expected runtime on a modern laptop: 45–120 seconds.**

You will see decreasing training and validation loss, followed by a sample of generated Shakespeare-style text.

### Advanced training

```bash
python train.py --max-iters 8000 --n-layer 6 --n-embd 192
```

## 2. Generate Text from the Command Line

```bash
# After training
python -m grokllm.generate --prompt "The prince said" --max-tokens 160 --temperature 0.8
```

Or with the entry point (after `pip install -e .`):

```bash
grokllm-generate -p "O happy dagger!" -m 120
```

## 3. Launch the Web Interface

```bash
python -m grokllm.web
# or simply
python app.py
```

Then open http://localhost:8000

The interface contains:
- Full sampling controls (temperature, top-k, top-p)
- Example prompts
- Prominent autonomous-build banners on every section
- Live performance stats

## 4. Use the Python API

```python
from grokllm.generate import GrokGenerator

gen = GrokGenerator.from_checkpoint("checkpoints/grokllm_tiny.pt")

result = gen.generate(
    prompt="To be or not",
    max_new_tokens=150,
    temperature=0.7,
    top_k=30
)

print(result["text"])
print(f"Speed: {result['tokens_per_second']} tok/s")
```

## 5. REST API

The server exposes a clean JSON API (see [API.md](API.md)).

Example with curl:

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The king entered the hall",
    "max_new_tokens": 100,
    "temperature": 0.85
  }'
```

## Common Workflows

**I just cloned the repo and want to play immediately (recommended)**

```bash
pip install -r requirements.txt
python train.py          # Train the model (~1 minute)
python app.py            # Start the web UI at http://localhost:8000
```

This is the most reliable path for new users.

**I want to experiment with a slightly larger model**

Edit `grokllm/config.py` or pass flags to `train.py`.

**I want to train on my own text**

Replace `data/tiny_shakespeare.txt` with any `.txt` file. The tokenizer and training loop are data-agnostic.

**I want to serve this somewhere**

The FastAPI app is production-ready enough for demos. Add a reverse proxy (Caddy / Nginx) + systemd if desired.

## Troubleshooting

| Problem                         | Solution |
|--------------------------------|----------|
| "No checkpoint found"          | Run `python train.py` |
| Very slow generation           | You're probably on CPU (expected). GPU helps little at this scale |
| Model outputs gibberish        | Try lowering temperature to 0.6–0.7 |
| Port 8000 already in use       | `uvicorn app:app --port 8001` |
| Out of memory                  | Reduce `--batch-size` during training |

## Attribution Reminder

At every step you will see messages like:

> "Built entirely by Grok Build CLI (YOLO mode)"

This is intentional. The agent was instructed to make the provenance unmistakable.

---

**All documentation and code in this repository was produced by Grok Build CLI.**
