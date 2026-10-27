"""Rank retrieved chunks with a cross-encoder."""
# Load .env before Hugging Face libraries read HF_ENDPOINT.
import config

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self):
        self.model = CrossEncoder(config.RERANKER_MODEL)

    def rerank(self, query: str, chunks: list[dict], top_k: int = config.FINAL_TOP_K) -> list[dict]:
        """Return the top_k chunks ranked by cross-encoder score."""
        if not chunks:
            return []
        scores = self.model.predict([(query, c["text"]) for c in chunks])
        scored = [{**c, "rerank_score": float(s)} for c, s in zip(chunks, scores)]
        return sorted(scored, key=lambda c: c["rerank_score"], reverse=True)[:top_k]


if __name__ == "__main__":
    import sys

    from retriever import Retriever

    question = sys.argv[1] if len(sys.argv) > 1 else "什么是混合检索"
    candidates = Retriever().search(question)
    for c in Reranker().rerank(question, candidates):
        print(f"[{c['rerank_score']:.3f}] {c['id']}\n  {c['text'][:60]}…\n")
