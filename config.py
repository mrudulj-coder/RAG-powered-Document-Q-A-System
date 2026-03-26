"""Centralized configuration for the RAG Document Q&A System."""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ── LLM Settings ─────────────────────────────────────────────────────
LLM_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/embedding-001"
TEMPERATURE = 0.3

# ── Chunking Settings ────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── ChromaDB Settings ────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
CHROMA_COLLECTION_NAME = "document_qa"

# ── Retrieval Settings ───────────────────────────────────────────────
TOP_K = 5
