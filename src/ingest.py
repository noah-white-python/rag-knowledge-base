"""Rebuild the local vector index and save chunks for BM25 retrieval."""
import json

# Load .env before Hugging Face libraries read HF_ENDPOINT.
import config

import chromadb
from sentence_transformers import SentenceTransformer

from chunker import chunk_documents
from loader import load_documents


def build_index():
    docs = load_documents()
    if not docs:
        print(f"data/ 目录下没有可用文档,请先放入 txt/md/pdf 文件:{config.DATA_DIR}")
        return
    chunks = chunk_documents(docs)
    if not chunks:
        print("切块结果为空(文档可能只有标题没有正文),没有可入库的内容")
        return
    print(f"\n共 {len(chunks)} 个文本块,开始向量化…")

    model = SentenceTransformer(config.EMBEDDING_MODEL)
    embeddings = model.encode(
        [c["text"] for c in chunks],
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks],
    )

    config.CHUNKS_FILE.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print(f"\n索引构建完成:{collection.count()} 块已入库 → {config.CHROMA_DIR}")


if __name__ == "__main__":
    build_index()
