"""
GrokLLM Command Line Interface

Entry point for `grokllm` command after installation.
All code written autonomously by Grok Build CLI.
"""

import argparse
import sys
import os


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    GROKLLM — THE SELF-BUILT LLM              ║
║                                                              ║
║   100% of this project was created by Grok Build CLI         ║
║   in YOLO mode. Zero human code. One prompt only.            ║
╚══════════════════════════════════════════════════════════════╝
""")


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        prog="grokllm",
        description="GrokLLM — the LLM built entirely by an AI coding agent"
    )
    subparsers = parser.add_subparsers(dest="command")

    # train
    p_train = subparsers.add_parser("train", help="Train the tiny Transformer")
    p_train.add_argument("--max-iters", type=int, default=None)

    # generate
    p_gen = subparsers.add_parser("generate", help="Generate text from a trained model")
    p_gen.add_argument("-p", "--prompt", default="To be or not to be")
    p_gen.add_argument("-m", "--max-tokens", type=int, default=150)
    p_gen.add_argument("-t", "--temperature", type=float, default=0.8)

    # serve
    p_serve = subparsers.add_parser("serve", help="Launch the web UI + API")
    p_serve.add_argument("--port", type=int, default=8000)

    # info
    subparsers.add_parser("info", help="Show attribution and build information")

    args = parser.parse_args()

    if args.command == "train":
        from .train import train as real_train
        from .config import GrokLLMConfig
        cfg = GrokLLMConfig()
        if args.max_iters:
            cfg.max_iters = args.max_iters
        real_train(cfg)

    elif args.command == "generate":
        from .generate import generate_text
        text = generate_text(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(text)

    elif args.command == "serve":
        import uvicorn
        import os
        import sys

        print(f"Starting GrokLLM web server on port {args.port}...")

        # Make sure 'app' module can be imported when running via installed script
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        uvicorn.run("app:app", host="0.0.0.0", port=args.port, reload=False)

    elif args.command == "info" or args.command is None:
        print("Project: GrokLLM")
        print("Built by: Grok Build CLI (Grok 4.3 by xAI)")
        print("Build date: April 2026")
        print("Human code edits: 0")
        print("Human steering allowed: only the word 'fix'")
        print()
        print("Quick commands:")
        print("  grokllm train")
        print("  grokllm generate -p 'The king'")
        print("  grokllm serve")
        print()
        print("See README.md for the full story.")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
