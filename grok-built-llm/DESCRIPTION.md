# GrokLLM — Repository Description

GrokLLM is a real, from-scratch decoder-only Transformer language model complete with training code, a FastAPI backend, a modern web interface, and comprehensive documentation. What makes the project remarkable is not its scale, but the fact that the entire repository was built autonomously.

In April 2026, Grok 4.3, operating through the Grok Build CLI, was given a single ambitious prompt: construct a simple but functional LLM with a web interface, write full documentation, and ensure it was unmistakably clear that no human had written any of the code. The agent ran in YOLO / always-approve mode. Human involvement was restricted to one-word commands such as "fix" when errors occurred. Several additional prompts were issued over the course of the session; every one of them, including this latest request, is recorded verbatim in CHALLENGE.md.

The technical output is a genuine ~820,000 parameter Transformer trained from random initialization on the Tiny Shakespeare dataset. On a modern laptop or Apple Silicon device, training completes in 45 to 90 seconds. The model supports temperature, top-k, and top-p sampling and is accessible through both a command-line generator and an attractive, self-contained web interface that requires no JavaScript framework or build step.

GrokLLM simultaneously serves as a clean educational implementation of a decoder-only Transformer, a fully runnable end-to-end demo, and a rare unfiltered artifact of frontier agentic software engineering. Every significant file contains explicit statements that it was written entirely by the Grok Build CLI with zero human code edits.

The project offers particular value to researchers studying autonomous coding systems, developers interested in minimal yet real LLM implementations, and anyone seeking to understand what AI agents can achieve with minimal supervision. The prompts that shaped it, the architectural decisions, the training dynamics, and the final product all coexist in one transparent repository.

To use it, clone the repository, install the dependencies, run `python train.py`, then start the interface with `python app.py`. You will be generating text from a neural network that quite literally created its own documentation, interface, and training story.

This is what building software looks like when the builder is no longer human.