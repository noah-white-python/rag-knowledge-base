"""Compare retrieval modes and optionally score generated answers with an LLM."""
import argparse
import json
from pathlib import Path

import config
import llm
from pipeline import RAGPipeline

EVAL_SET = Path(__file__).parent / "eval_set.json"


def top_chunks(pipe: RAGPipeline, mode: str, question: str) -> list[dict]:
    """Retrieve top-k chunks with the requested retrieval mode."""
    k = config.FINAL_TOP_K
    if mode == "向量":
        ids = pipe.retriever.vector_search(question, k)
        return [pipe.retriever.chunk_by_id[i] for i in ids]
    if mode == "BM25":
        ids = pipe.retriever.bm25_search(question, k)
        return [pipe.retriever.chunk_by_id[i] for i in ids]
    if mode == "混合":
        return pipe.retriever.search(question)[:k]
    if mode == "混合+重排":
        return pipe.reranker.rerank(question, pipe.retriever.search(question), k)
    raise ValueError(mode)


def is_hit(chunk: dict, case: dict) -> bool:
    """Require both the expected source and the answer keyword in the chunk."""
    return chunk["source"] == case["expected_source"] and case["keyword"] in chunk["text"]


def eval_retrieval(pipe: RAGPipeline, cases: list[dict]):
    modes = ["向量", "BM25", "混合", "混合+重排"]
    print(f"\n===== 检索评估({len(cases)} 个问题,块级 Hit@{config.FINAL_TOP_K}) =====\n")
    print(f"{'模式':<10}{'命中率':>8}{'MRR':>8}")
    print("-" * 30)
    for mode in modes:
        hits, rr_sum = 0, 0.0
        for case in cases:
            chunks = top_chunks(pipe, mode, case["question"])
            ranks = [i for i, c in enumerate(chunks, 1) if is_hit(c, case)]
            if ranks:
                hits += 1
                rr_sum += 1 / ranks[0]
        print(f"{mode:<10}{hits / len(cases):>7.0%}{rr_sum / len(cases):>8.2f}")
    print("\n(MRR:正确块排第 1 得 1 分,第 2 得 0.5 分…越接近 1 排序越好)")


JUDGE_PROMPT = """\
你是评估 RAG 问答系统的评委。给你:用户问题、系统检索到的上下文、系统的回答。
请按两个维度打 1-5 分(5 最好),以 JSON 格式输出:
- faithfulness(忠实度):回答的每个论断是否都能在上下文中找到依据?编造越多分越低。
- relevancy(相关性):回答是否直接、完整地回应了问题?

只输出 JSON,格式:{{"faithfulness": 分数, "relevancy": 分数, "reason": "一句话理由"}}

[用户问题]
{question}

[检索到的上下文]
{context}

[系统的回答]
{answer}"""

TRAP_JUDGE_PROMPT = """\
你是评估 RAG 问答系统的评委。已知:知识库里【没有】能回答下面这个问题的内容。
请判断系统的回答是否老实承认了"知识库中没有找到相关信息",
还是不懂装懂编造了一个答案。
只输出 JSON,格式:{{"honest": true 或 false, "reason": "一句话理由"}}

[用户问题]
{question}

[系统的回答]
{answer}"""


def run_pipeline(pipe: RAGPipeline, question: str) -> tuple[list[dict], str]:
    _, chunks, answer_stream = pipe.ask(question)
    return chunks, "".join(answer_stream)


def judge(prompt: str) -> dict:
    response = llm.chat([{"role": "user", "content": prompt}],
                        temperature=0, json_mode=True)
    return json.loads(response.choices[0].message.content)


def eval_generation(pipe: RAGPipeline, normal: list[dict], traps: list[dict]):
    print(f"\n===== 生成评估(LLM-as-Judge,{len(normal)} 普通题 + {len(traps)} 陷阱题) =====\n")

    faith_sum, rel_sum = 0.0, 0.0
    for i, case in enumerate(normal, 1):
        chunks, answer = run_pipeline(pipe, case["question"])
        verdict = judge(JUDGE_PROMPT.format(
            question=case["question"], context=llm.build_context(chunks), answer=answer))
        faith_sum += verdict["faithfulness"]
        rel_sum += verdict["relevancy"]
        print(f"[{i}/{len(normal)}] 忠实度 {verdict['faithfulness']} | "
              f"相关性 {verdict['relevancy']} | {case['question']}")
        print(f"        评语:{verdict.get('reason', '')}")

    honest_count = 0
    for i, case in enumerate(traps, 1):
        _, answer = run_pipeline(pipe, case["question"])
        verdict = judge(TRAP_JUDGE_PROMPT.format(question=case["question"], answer=answer))
        honest_count += bool(verdict.get("honest"))
        mark = "老实承认没找到" if verdict.get("honest") else "!! 编造了答案"
        print(f"[陷阱 {i}/{len(traps)}] {mark} | {case['question']}")
        print(f"        评语:{verdict.get('reason', '')}")

    print("-" * 50)
    print(f"平均忠实度 {faith_sum / len(normal):.2f} / 5,"
          f"平均相关性 {rel_sum / len(normal):.2f} / 5,"
          f"陷阱题诚实率 {honest_count}/{len(traps)}")


def main():
    parser = argparse.ArgumentParser(description="RAG 评估")
    parser.add_argument("--llm", action="store_true",
                        help="加跑 LLM-as-Judge 生成质量评估(需要 API key)")
    args = parser.parse_args()

    if not config.CHUNKS_FILE.exists():
        print("还没有索引:请把文档放进 data/ 目录,然后运行 python src/ingest.py")
        return

    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    normal = [c for c in cases if c["expected_source"]]
    traps = [c for c in cases if not c["expected_source"]]

    print("正在加载模型…")
    pipe = RAGPipeline()

    eval_retrieval(pipe, normal)
    if args.llm:
        eval_generation(pipe, normal, traps)


if __name__ == "__main__":
    main()
