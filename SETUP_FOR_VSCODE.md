# Complete AI Agent Setup Guide for VS Code

This guide provides everything needed to replicate the **Direct Agricultural AI Agent** in another VS Code instance using Copilot.

---

## Overview

The AI Agent is a direct Ollama-based agricultural assistant that:
- Uses only approved Ollama models (no RAG or retrieval)
- Maintains conversation history in memory
- Provides answers through direct reasoning
- Saves session memory automatically
- Runs locally with no external dependencies

---

## Prerequisites

Before setting up, ensure you have:

1. **Python 3.8+** installed
2. **Ollama** installed from https://ollama.ai
3. **VS Code** with GitHub Copilot access
4. **Git** for version control (optional)

---

## Step 1: Create Project Directory

```bash
# Create the main project folder
mkdir ai_agent_demo
cd ai_agent_demo
```

---

## Step 2: Create Python Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

---

## Step 3: Create Project Files

Create the following files in the `ai_agent_demo` directory:

### 3.1 requirements.txt

```
ollama
pypdf
pymupdf
pytesseract
Pillow
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3.2 config.py

```python
import os

AVAILABLE_MODELS = [
    "lfm2.5:1.2b",
    "phi4-mini:3.8b",
    "qwen3:0.6b",
    "qwen3:4b",
    "qwen2.5:1.5b",
]

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "phi4-mini:3.8b")
MODEL_SELECTION = os.getenv("AGRI_MODEL_SELECTION", DEFAULT_MODEL)

SYSTEM_PROMPT = (
    "You are a direct, careful agricultural AI assistant. "
    "Answer from your own reasoning without retrieving documents or pretending to cite sources. "
    "If the user asks for a fact you are uncertain about, say so clearly and offer the safest next step. "
    "Prefer practical, concise, high-signal answers with short structure when it helps clarity."
)

# Optional OCR configuration (used in tools.py)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
ENABLE_OCR = False
OCR_DPI = 300
OCR_LANG = "eng"
```

### 3.3 memory.py

```python
from datetime import datetime
import json
from pathlib import Path


class Memory:
    def __init__(self):
        self.messages = []

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})

    def get_all(self):
        return self.messages

    def clear(self):
        self.messages.clear()

    def to_markdown(self):
        lines = ["# Session Memory", ""]
        for message in self.messages:
            role = message.get("role", "unknown").title()
            content = message.get("content", "")
            lines.append(f"## {role}")
            lines.append(content)
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def save(self, output_dir="reports"):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        markdown_path = path / f"session_memory_{timestamp}.md"
        json_path = path / f"session_memory_{timestamp}.json"

        markdown_path.write_text(self.to_markdown(), encoding="utf-8")
        json_path.write_text(json.dumps(self.messages, indent=2, ensure_ascii=False), encoding="utf-8")

        return markdown_path, json_path
```

### 3.4 agent.py

```python
import subprocess

import ollama

from config import AVAILABLE_MODELS, MODEL_SELECTION, SYSTEM_PROMPT
from memory import Memory


def _available_ollama_models():
    try:
        proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=8)
    except Exception:
        return []

    if proc.returncode != 0:
        return []

    models = []
    for line in proc.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models


def _select_model(preferred_model: str):
    if preferred_model not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{preferred_model}' is not in the approved shortlist.")

    installed_models = _available_ollama_models()
    if installed_models and preferred_model not in installed_models:
        for model_name in AVAILABLE_MODELS:
            if model_name in installed_models:
                return model_name

    return preferred_model


class AIAgent:
    def __init__(self, model_name: str = MODEL_SELECTION):
        self.memory = Memory()
        self.model_name = _select_model(model_name)

    def build_messages(self, user_input):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        messages.extend(self.memory.get_all())
        return messages

    def chat(self, user_input):
        self.memory.add("user", user_input)

        try:
            result = ollama.chat(
                model=self.model_name,
                messages=self.build_messages(user_input),
            )
            response = result["message"]["content"]
        except Exception as exc:
            response = (
                "Data not found: the selected Ollama model is not available or the Ollama service is not ready. "
                f"Details: {exc}"
            )

        self.memory.add("assistant", response)
        return response

    def research_brief(self):
        return f"Model: {self.model_name}\nMode: direct chat, no RAG"

    def _looks_like_not_found(self, answer: str):
        normalized = (answer or "").strip().lower()
        if not normalized:
            return True

        not_found_markers = (
            "i don't know",
            "i do not know",
            "not found",
            "no data",
            "cannot find",
            "can't find",
            "no relevant",
            "not available",
            "unable to",
        )
        return any(marker in normalized for marker in not_found_markers)

    def generate_report(self, user_input):
        answer = self.chat(user_input)
        if self._looks_like_not_found(answer):
            return (
                f"# Data Not Found Report\n\n"
                f"## Question\n{user_input}\n\n"
                f"## Status\nData not found or not confidently available from the model response.\n\n"
                f"## Response\n{answer}\n\n"
                f"## Model\n{self.model_name}\n"
            )

        return (
            f"# AI Agent Response\n\n"
            f"## Question\n{user_input}\n\n"
            f"## Answer\n{answer}\n\n"
            f"## Model\n{self.model_name}\n"
        )
```

### 3.5 main.py

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import argparse
from datetime import datetime
from pathlib import Path

from agent import AIAgent
from config import AVAILABLE_MODELS, DEFAULT_MODEL


def parse_args():
    parser = argparse.ArgumentParser(description="Direct agricultural AI chat agent")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=AVAILABLE_MODELS,
        help="Approved Ollama model to use",
    )
    parser.add_argument(
        "--ask",
        default=None,
        help="Ask one question and exit instead of starting the interactive loop",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    agent = AIAgent(model_name=args.model)
    session_saved = False

    def save_session_memory():
        nonlocal session_saved
        if session_saved:
            return

        markdown_path, json_path = agent.memory.save()
        print(f"Session memory saved to {markdown_path}")
        print(f"Session memory JSON saved to {json_path}")
        session_saved = True

    print("Direct AI Agent Started (type 'exit' to stop)\n")
    print(agent.research_brief() + "\n")
    print("Ask for crop advice, comparisons, summaries, drafting help, or step-by-step reasoning.\n")

    if args.ask:
        print(agent.generate_report(args.ask))
        save_session_memory()
        return

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            save_session_memory()
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            save_session_memory()
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            agent.memory.clear()
            print("Chat history cleared.\n")
            continue

        if user_input.lower() == "brief":
            print(agent.research_brief() + "\n")
            continue

        print("\nAssistant:", agent.chat(user_input), "\n")


if __name__ == "__main__":
    main()
```

### 3.6 tools.py (Optional - for advanced features)

```python
from pathlib import Path
import re
import io
from collections import Counter

from config import CHUNK_OVERLAP, CHUNK_SIZE, ENABLE_OCR, OCR_DPI, OCR_LANG
from pypdf import PdfReader

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import fitz
except Exception:
    fitz = None

try:
    import pytesseract
except Exception:
    pytesseract = None


def calculator(expression: str):
    try:
        return str(eval(expression))
    except Exception:
        return "Invalid expression"


def _normalize_text(text: str):
    return re.sub(r"\s+", " ", text or "").strip()


def _tokenize(text: str):
    return [token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(token) > 2]


def _chunk_text(text: str, chunk_size: int, overlap: int):
    cleaned = _normalize_text(text)
    if not cleaned:
        return []

    chunks = []
    start = 0
    text_length = len(cleaned)

    while start < text_length:
        end = min(text_length, start + chunk_size)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_length:
            break
        start = max(end - overlap, start + 1)

    return chunks


def _read_pdf_pages(file_path: str):
    reader = PdfReader(file_path)
    pages = []

    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page": index, "text": text.strip()})

    return pages


def _ocr_pdf_pages(file_path: str):
    if not ENABLE_OCR or fitz is None or pytesseract is None or Image is None:
        return []

    pages = []
    document = fitz.open(file_path)

    try:
        for index, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(dpi=OCR_DPI)
            image_bytes = pixmap.tobytes("png")
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image, lang=OCR_LANG)
            pages.append({"page": index, "text": _normalize_text(text), "ocr": True})
    finally:
        document.close()

    return pages


def _iter_pdf_files(path: Path):
    if path.is_file():
        if path.suffix.lower() == ".pdf":
            yield path
        return

    if path.is_dir():
        for file_path in sorted(path.rglob("*.pdf")):
            yield file_path
```

### 3.7 .gitignore (Optional but recommended)

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Reports
reports/

# Environment
.env
.env.local
```

---

## Step 4: Prepare Ollama Models

Before running the agent, download an approved model:

```bash
# Start Ollama service
ollama serve

# In another terminal, download a model
ollama pull phi4-mini:3.8b

# Or try other approved models:
ollama pull lfm2.5:1.2b
ollama pull qwen2.5:1.5b
```

Approved models:
- `phi4-mini:3.8b` (recommended - smallest and fastest)
- `lfm2.5:1.2b`
- `qwen2.5:1.5b`
- `qwen3:0.6b`
- `qwen3:4b`

---

## Step 5: Run the Agent

### Interactive Chat Mode
```bash
python main.py
```

Commands available in chat:
- `exit` - Save session and quit
- `reset` - Clear chat history
- `brief` - Show current model and mode

### Ask a Single Question
```bash
python main.py --ask "What are the best crops for sandy soil?"
```

### Use a Specific Model
```bash
python main.py --model "lfm2.5:1.2b"
```

---

## Step 6: Environment Variables (Optional)

For customization, set these environment variables:

```bash
# Windows PowerShell
$env:OLLAMA_MODEL = "lfm2.5:1.2b"
$env:AGRI_MODEL_SELECTION = "phi4-mini:3.8b"

# Windows CMD
set OLLAMA_MODEL=lfm2.5:1.2b
set AGRI_MODEL_SELECTION=phi4-mini:3.8b

# macOS/Linux
export OLLAMA_MODEL="lfm2.5:1.2b"
export AGRI_MODEL_SELECTION="phi4-mini:3.8b"
```

---

## Step 7: VS Code Integration with Copilot

### Add the Following to VS Code Settings

In VS Code, open settings (`Ctrl+,` or `Cmd+,`) and add:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.linting.enabled": true,
    "python.formatting.provider": "black",
    "[python]": {
        "editor.formatOnSave": true
    },
    "terminal.integrated.python.executeInFileDir": true
}
```

### Add Tasks for Quick Execution

Create `.vscode/tasks.json`:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Run AI Agent (Interactive)",
            "type": "shell",
            "command": "${command:python.interpreterPath}",
            "args": ["main.py"],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "problemMatcher": ["$python"],
            "cwd": "${workspaceFolder}"
        },
        {
            "label": "Run AI Agent (Single Question)",
            "type": "shell",
            "command": "${command:python.interpreterPath}",
            "args": ["main.py", "--ask", "What is the best crop for this season?"],
            "group": "build",
            "problemMatcher": ["$python"],
            "cwd": "${workspaceFolder}"
        },
        {
            "label": "Install Dependencies",
            "type": "shell",
            "command": "${command:python.interpreterPath}",
            "args": ["-m", "pip", "install", "-r", "requirements.txt"],
            "group": "build",
            "problemMatcher": ["$python"],
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

### Add Debug Configuration

Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: AI Agent",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main.py",
            "console": "integratedTerminal",
            "justMyCode": true
        },
        {
            "name": "Python: Agent with Question",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main.py",
            "args": ["--ask", "How do I grow tomatoes?"],
            "console": "integratedTerminal",
            "justMyCode": true
        }
    ]
}
```

---

## File Structure After Setup

```
ai_agent_demo/
├── venv/                    # Virtual environment
├── reports/                 # Session memory saves
├── .vscode/
│   ├── tasks.json          # VS Code tasks
│   └── launch.json         # Debug configuration
├── agent.py                # AI Agent class
├── config.py               # Configuration
├── main.py                 # Entry point
├── memory.py               # Conversation memory
├── tools.py                # Optional tools (OCR, PDF)
├── requirements.txt        # Dependencies
├── .gitignore             # Git ignore file
└── SETUP_FOR_VSCODE.md    # This file
```

---

## Troubleshooting

### Issue: "Ollama service not running"
**Solution:** Start Ollama:
```bash
ollama serve
```

### Issue: "Model not found"
**Solution:** Pull the model:
```bash
ollama pull phi4-mini:3.8b
```

### Issue: Python venv not activating
**Solution (Windows):**
```bash
venv\Scripts\activate
```

**Solution (macOS/Linux):**
```bash
source venv/bin/activate
```

### Issue: pytesseract error
**Solution:** The OCR feature is optional. If you don't need it, set `ENABLE_OCR = False` in `config.py`.

### Issue: Permission denied on macOS/Linux
**Solution:** Make the script executable:
```bash
chmod +x main.py
```

---

## Using with Copilot in VS Code

1. **Open the project folder** in VS Code
2. **Use Copilot Chat** (`Ctrl+Shift+I` or `Cmd+Shift+I`)
3. **Ask questions like:**
   - "What does the AIAgent class do?"
   - "How do I add a new model to the approved list?"
   - "Explain the memory management system"
   - "How to integrate this with a database?"

4. **Copilot will help you:**
   - Understand the codebase
   - Extend functionality
   - Debug issues
   - Optimize performance

---

## Quick Start Summary

```bash
# 1. Create and enter project directory
mkdir ai_agent_demo && cd ai_agent_demo

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Create all the Python files (config.py, memory.py, agent.py, main.py, tools.py)

# 4. Create requirements.txt with dependencies

# 5. Install dependencies
pip install -r requirements.txt

# 6. Start Ollama (in separate terminal)
ollama serve

# 7. Pull a model
ollama pull phi4-mini:3.8b

# 8. Run the agent
python main.py
```

---

## Advantages of This Setup

✅ **Lightweight** - No external APIs or cloud dependencies
✅ **Private** - All data stays local
✅ **Fast** - Direct reasoning without retrieval overhead
✅ **Extensible** - Easy to add custom models or features
✅ **Reproducible** - Same setup works across machines
✅ **Copilot-friendly** - Clean code for AI assistance

---

## Next Steps

1. Replicate this exact structure in your new VS Code workspace
2. Follow the steps in the "Quick Start Summary"
3. Test with the interactive chat mode
4. Use Copilot to extend and customize features

Good luck with your new AI Agent setup!
