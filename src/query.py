"""Command-line chat interface."""
import argparse
import sys

import config
import llm
import pipeline
from pipeline import RAGPipeline

# Allow redirected output on Windows consoles with limited encodings.
sys.stdout.reconfigure(errors="replace")


def print_sources(chunks: list[dict], verbose: bool):
    print("\n" + "-" * 40)
    print("参考来源:")
    for i, c in enumerate(chunks, 1):
        score = f"  (重排分 {c['rerank_score']:.3f})" if "rerank_score" in c else ""
        print(f"  [{i}] {c['source']} 第{c['chunk_index']}块{score}")
        if verbose:
            print(f"      {c['text'][:120]}…")


def stream_to_console(answer) -> str:
    """Print streamed text and return the complete answer."""
    full_answer = ""
    for piece in answer:
        print(piece, end="", flush=True)
        full_answer += piece
    print()
    return full_answer


def ask_once(pipe: RAGPipeline, question: str, history: list[dict],
             verbose: bool, use_crag: bool = False, use_check: bool = False) -> str:
    search_query, chunks, answer = pipe.ask(question, history, use_crag=use_crag)
    if search_query != question:
        print(f"(检索查询已改写为:{search_query})")
    if not chunks:
        print("知识库中没有找到相关内容,请先运行 python src/ingest.py 构建索引。")
        return ""
    if pipeline.low_confidence(chunks):
        print("[警告] 检索到的内容与问题相关性都很低,以下回答可能不可靠:\n")

    full_answer = stream_to_console(answer)

    if use_check:
        verdict = llm.verify_answer(question, chunks, full_answer)
        if not verdict.get("supported", True):
            print(f"\n[自检] 发现可能无依据的内容({verdict.get('reason', '')}),重新生成:")
            full_answer = stream_to_console(
                llm.answer_stream(question, chunks, history, strict=True)
            )

    print_sources(chunks, verbose)
    return full_answer


def main():
    parser = argparse.ArgumentParser(description="RAG 知识库问答")
    parser.add_argument("question", nargs="?", help="要问的问题;不传则进入交互模式")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示检索到的块内容")
    parser.add_argument("--no-rerank", action="store_true", help="关闭 cross-encoder 重排")
    parser.add_argument("--crag", action="store_true",
                        help="开启 CRAG 自纠错:LLM 过滤不相关块,必要时改写查询重试")
    parser.add_argument("--check", action="store_true",
                        help="生成后自检回答是否有据可依,不合格自动重写一次")
    args = parser.parse_args()

    if not config.DEEPSEEK_API_KEY or config.DEEPSEEK_API_KEY == "sk-your-key-here":
        print("错误:没有配置 DEEPSEEK_API_KEY,请编辑项目根目录的 .env 文件填入你的 key")
        return
    if not config.CHUNKS_FILE.exists():
        print("还没有索引:请把文档放进 data/ 目录,然后运行 python src/ingest.py")
        return

    print("正在加载模型…")
    pipe = RAGPipeline(use_reranker=not args.no_rerank)

    if args.question:
        ask_once(pipe, args.question, history=[], verbose=args.verbose,
                 use_crag=args.crag, use_check=args.check)
        return

    print("已就绪,输入问题开始对话(q 退出)\n")
    history: list[dict] = []
    while True:
        try:
            question = input("你:").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in ("q", "quit", "exit"):
            break
        print("助手:", end="")
        answer = ask_once(pipe, question, history, args.verbose, args.crag, args.check)
        if answer:
            history += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        print()


if __name__ == "__main__":
    main()
