"""
Web server launcher for `python -m grokllm.web` and `grokllm-serve`

Written entirely by Grok Build CLI.

This module supports two ways of running:
- From the source directory: `python -m grokllm.web` or `python app.py`
- After `pip install -e .`: `grokllm-serve`
"""

import os
import sys


def main():
    print("Launching GrokLLM web interface...")
    print("Open http://localhost:8000 in your browser")
    print()

    # Ensure we can import 'app' when running as an installed package
    # from the project root (the most common and supported way)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        import uvicorn
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
    except ImportError:
        print("Error: uvicorn not found. Please run: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
