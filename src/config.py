"""
Central configuration for DocuChat.
Reads overrides from environment variables (loaded from .env via python-dotenv),
falling back to sensible defaults so the app works out of the box.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads .env if present; safe no-op otherwise

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DOCS_DIR = PROJECT_ROOT / "data" / "raw_docs"
STORAGE_DIR = PROJECT_ROOT / "storage"
FAISS_INDEX_PATH = STORAGE_DIR / "faiss_index" / "index.faiss"
METADATA_PATH = STORAGE_DIR / "metadata.pkl"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)

# --- API keys ------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- Models ----------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --- Chunking ------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))

# --- Retrieval -----------------------------------------------------------
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", 10))  # candidates from FAISS
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 4))       # final passages after reranking
