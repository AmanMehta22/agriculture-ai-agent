import os

AVAILABLE_MODELS = [
    "lfm2.5:1.2b",
    "phi4-mini:3.8b",
    "qwen3:0.6b",
    "qwen3:4b",
    "qwen2.5:1.5b",
    "llama3:latest",
]

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")
MODEL_SELECTION = os.getenv("AGRI_MODEL_SELECTION", DEFAULT_MODEL)

# Knowledge Base Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
ENABLE_OCR = False
OCR_DPI = 300
OCR_LANG = 'eng'

SYSTEM_PROMPT = (
    "You are a direct, careful agricultural AI assistant. "
    "Answer from your own reasoning without retrieving documents or pretending to cite sources. "
    "If the user asks for a fact you are uncertain about, say so clearly and offer the safest next step. "
    "Prefer practical, concise, high-signal answers with short structure when it helps clarity."
)
