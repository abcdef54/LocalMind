import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

UPLOAD_DIR = PROJECT_ROOT / '.uploads'

VECTOR_DB_PERSIST_DIR = UPLOAD_DIR / 'chroma'


RECENT_MESSAGES_COUNT = 20
SUMMARY_BATCH = RECENT_MESSAGES_COUNT

LLAMA_SWAP_CONFIG_PATH = PROJECT_ROOT / os.getenv("LLAMA_SWAP_CONFIG_PATH", "llama_swap_config.yaml")
LLAMA_SWAP_PATH_PREFIX = os.getenv("LLAMA_SWAP_PATH_PREFIX", "/models/")
DEFAULT_CONTEXT_LENGTH_K = int(os.getenv("DEFAULT_CONTEXT_LENGTH_K", "32"))

DEFAULT_MODEL_REPO_ID = "Qwen/Qwen2.5-3B-Instruct-GGUF"
DEFAULT_MODEL_GGUF_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"
