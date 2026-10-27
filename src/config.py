"""Paths, model names, and retrieval settings."""
import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"
CHUNKS_FILE = CHROMA_DIR / "chunks.json"


load_dotenv(ROOT_DIR / ".env")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")


EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"

LLM_MODEL = "deepseek-v4-flash"
LLM_BASE_URL = "https://api.deepseek.com"


# Apply the instruction to queries only, not document embeddings.
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


# Body length in characters; source and heading prefixes are added afterward.
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80


COLLECTION_NAME = "knowledge_base"
VECTOR_TOP_K = 20
BM25_TOP_K = 20
RRF_K = 60
RERANK_CANDIDATES = 20
FINAL_TOP_K = 5
# A heuristic cutoff, not a calibrated probability.
LOW_RELEVANCE_THRESHOLD = 0.3


LLM_TEMPERATURE = 0.2
MAX_HISTORY_TURNS = 3
