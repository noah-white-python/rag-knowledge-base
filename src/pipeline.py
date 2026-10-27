"""Shared retrieval and answer pipeline for the CLI and web interface."""
import config
import llm
from reranker import Reranker
from retriever import Retriever


def low_confidence(chunks: list[dict]) -> bool:
    """Check the highest reranker score against the configured threshold."""
    return bool(chunks) and chunks[0].get("rerank_score", 1.0) < config.LOW_RELEVANCE_THRESHOLD


class RAGPipeline:
    def __init__(self, use_reranker: bool = True):
        self.retriever = Retriever()
        self.reranker = Reranker() if use_reranker else None

    def retrieve(self, query: str, top_k: int = config.FINAL_TOP_K,
                 use_reranker: bool = True) -> list[dict]:
        """Retrieve candidates and optionally rerank them before selecting top_k."""
        candidates = self.retriever.search(query)
        if self.reranker and use_reranker:
            return self.reranker.rerank(query, candidates, top_k)
        return candidates[:top_k]

    def crag_filter(self, query: str, chunks: list[dict], top_k: int,
                    use_reranker: bool) -> tuple[list[dict], str | None]:
        """Filter irrelevant chunks; rewrite and retry once if none remain."""
        keep = llm.grade_chunks(query, chunks)
        good = [c for c, ok in zip(chunks, keep) if ok]
        if good:
            return good, None
        rewritten = llm.rewrite_query(query)
        return self.retrieve(rewritten, top_k, use_reranker), rewritten

    def ask(self, question: str, history: list[dict] | None = None,
            top_k: int = config.FINAL_TOP_K, use_reranker: bool = True,
            use_crag: bool = False):
        """Return the search query, selected chunks, and an answer iterator."""
        history = history or []
        search_query = llm.condense_question(history, question)
        chunks = self.retrieve(search_query, top_k, use_reranker)
        if use_crag and chunks:
            chunks, rewritten = self.crag_filter(search_query, chunks, top_k, use_reranker)
            if rewritten:
                search_query = rewritten
        answer = llm.answer_stream(question, chunks, history)
        return search_query, chunks, answer
