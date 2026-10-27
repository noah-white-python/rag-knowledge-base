"""Vector and BM25 retrieval combined with reciprocal rank fusion."""
import json
import re

# Load .env before Hugging Face libraries read HF_ENDPOINT.
import config

import chromadb
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


STOPWORDS = {
    "的", "了", "是", "在", "和", "有", "就", "不", "都", "而", "及", "与", "这", "那",
    "个", "中", "也", "很", "到", "要", "会", "着", "被", "把", "让", "向", "于", "并",
    "等", "对", "从", "它", "们", "上", "下", "为", "以", "或", "如", "但", "其", "之",
    "一个", "我们", "可以", "什么", "怎么", "如果", "因为", "所以", "比如", "通常",
}


def tokenize(text: str) -> list[str]:
    """Tokenize Chinese text and remove stopwords and punctuation."""
    return [
        token for token in jieba.cut(text)
        if token.strip() and token not in STOPWORDS and re.search(r"\w", token)
    ]


class Retriever:
    def __init__(self):
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection = client.get_collection(config.COLLECTION_NAME)

        self.chunks = json.loads(config.CHUNKS_FILE.read_text(encoding="utf-8"))
        self.chunk_by_id = {c["id"]: c for c in self.chunks}
        corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)

    def vector_search(self, query: str, k: int = config.VECTOR_TOP_K) -> list[str]:
        """Return chunk IDs ordered by vector similarity."""
        embedding = self.embedder.encode(
            config.BGE_QUERY_INSTRUCTION + query,
            normalize_embeddings=True,
        )
        k = min(k, self.collection.count())
        result = self.collection.query(query_embeddings=[embedding.tolist()], n_results=k)
        return result["ids"][0]

    def bm25_search(self, query: str, k: int = config.BM25_TOP_K) -> list[str]:
        """Return chunk IDs with positive BM25 scores, highest first."""
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i]["id"] for i in top if scores[i] > 0]

    def search(self, query: str) -> list[dict]:
        """Fuse vector and BM25 rankings with RRF."""
        routes = [self.vector_search(query), self.bm25_search(query)]

        rrf_scores: dict[str, float] = {}
        for ids in routes:
            for rank, chunk_id in enumerate(ids):
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (
                    config.RRF_K + rank + 1
                )
        fused_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        return [
            {**self.chunk_by_id[i], "rrf_score": rrf_scores[i]}
            for i in fused_ids[: config.RERANK_CANDIDATES]
        ]


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "什么是混合检索"
    retriever = Retriever()
    for c in retriever.search(question)[:5]:
        print(f"[{c['rrf_score']:.4f}] {c['id']}\n  {c['text'][:60]}…\n")
