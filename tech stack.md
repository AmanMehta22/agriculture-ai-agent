# Tech Stack

## Core Language
- Python

## AI / LLM
- Ollama for local model serving
- Approved Ollama models from `config.py`
- `phi4-mini:3.8b`
- `llama3:latest`
- `qwen3:0.6b`
- `qwen3:4b`
- `qwen2.5:1.5b`
- `lfm2.5:1.2b`

## Application Runtime
- Command-line interface built with `argparse`
- Local chat loop and one-shot question mode
- Conversation state kept in memory

## Knowledge Base / Data Layer
- Local POP Bank knowledge base stored in `POP Bank_Kshitij/`
- PDF index stored in `pop_bank_index.json`
- Extracted text cache stored in `pop_bank_cache.json`
- On-demand state/crop loading through `pop_bank_loader.py`

## Document Processing
- `pypdf`
- `pymupdf`
- `pytesseract`
- `Pillow`

## Project Modules
- `main.py` for CLI entry point
- `agent.py` for response generation and knowledge lookup
- `memory.py` for chat history persistence
- `tools.py` for PDF/OCR helper utilities
- `config.py` for model and knowledge-base settings

## Storage
- JSON files for cache and index
- Markdown and JSON session reports in `reports/`
- Local filesystem only

## Platform
- Windows
- Local development in VS Code

## Notes
- The app is currently a direct local agent, not a web app.
- The POP Bank state answers are generated from the local indexed PDFs and cache.
