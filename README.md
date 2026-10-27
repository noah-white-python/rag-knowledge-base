# RAG Knowledge Base

A local document question-answering app with a Streamlit interface and a CLI.
It indexes text, Markdown, and text-based PDFs, then sends retrieved passages to
DeepSeek to generate answers with numbered source references.

Retrieval combines ChromaDB vector search and BM25, merges the rankings with RRF,
and optionally reranks candidates with a cross-encoder. Follow-up questions are
rewritten using the recent conversation history.

## Setup

Use Python 3.12. From the project directory on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `DEEPSEEK_API_KEY` in `.env`. The model name and API URL are in
`src/config.py`; choose a model available to your account.

Put documents in `data/`, including subdirectories, then run:

```powershell
python src/ingest.py
streamlit run src/app.py
```

The first index build downloads `BAAI/bge-small-zh-v1.5`. Queries also load
`BAAI/bge-reranker-base` unless reranking is disabled in the CLI. Model downloads
require network access. The optional `HF_ENDPOINT` setting is in `.env.example`.

Indexing runs locally. Answer generation, follow-up rewriting, and optional LLM
checks send questions and relevant text to the configured API. Only index
material you are permitted to send there.

## CLI

```powershell
python src/query.py "什么是混合检索？"
python src/query.py
python src/query.py -v "RRF 的公式是什么？"
python src/query.py --no-rerank "什么是重排序？"
python src/query.py --crag "什么是查询改写？"
python src/query.py --check "BM25 的局限是什么？"
```

Without a question, the CLI starts an interactive conversation; enter `q` to exit.
`-v` prints retrieved text excerpts. `--crag` asks the LLM to filter passages and
rewrites the query once if all are rejected. The retry results are not graded
again. `--check` checks the generated answer against its context and regenerates
it once if the check fails. Both options add API calls.

## Evaluation

```powershell
python src/eval.py
python src/eval.py --llm
python -m unittest discover -s tests
```

The retrieval evaluation compares vector search, BM25, hybrid search, and hybrid
search with reranking. A hit requires the expected source and a keyword in the
retrieved chunk. It reports Hit@5 and reciprocal rank within the top five results.
`--llm` also generates answers and asks the same configured model to score them.

The bundled set contains 12 answerable questions and 3 questions outside the
sample documents. It is a small smoke test, not a benchmark or evidence that one
retrieval method consistently outperforms another. Update the cases when using
your own documents.

## Files

| Path | Purpose |
| --- | --- |
| `src/config.py` | Paths, models, and retrieval parameters |
| `src/loader.py`, `src/chunker.py`, `src/ingest.py` | Load documents and rebuild the index |
| `src/retriever.py`, `src/reranker.py` | Retrieve and rank passages |
| `src/llm.py`, `src/pipeline.py` | API calls and shared question-answering flow |
| `src/app.py`, `src/query.py` | Web and CLI entry points |
| `src/eval.py`, `src/eval_set.json` | Evaluation runner and sample cases |
| `data/` | Three Chinese example documents |
| `tests/` | Regression checks without model downloads or API calls |

## Limitations

- `ingest.py` replaces the existing collection. Rebuild after changing documents,
  embedding models, or chunk settings. Restart the web app or use its reload
  button after rebuilding.
- PDFs must contain extractable text; OCR is not included.
- Chunk size is measured in characters, before source and heading prefixes.
  It does not guarantee a token limit for the embedding model.
- Numbered citations and LLM checks do not guarantee factual accuracy. The
  relevance threshold is a heuristic, not a confidence probability.
- The web interface has no authentication or per-user document isolation.

`.env`, virtual environments, IDE settings, and `chroma_db/` are excluded from
Git. Review any documents you add under `data/` before publishing them.
