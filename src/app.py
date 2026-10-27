"""Streamlit chat interface."""
import streamlit as st

import config
import llm
import pipeline as pipeline_mod
from pipeline import RAGPipeline

st.set_page_config(page_title="RAG 知识库问答", layout="centered")


@st.cache_resource(show_spinner="首次启动:正在加载 embedding 和重排模型…")
def load_pipeline() -> RAGPipeline:
    return RAGPipeline()


try:
    pipe = load_pipeline()
except Exception as e:
    st.error(f"初始化失败:{e}\n\n请先运行 `python src/ingest.py` 构建索引。")
    st.stop()


with st.sidebar:
    st.title("RAG 知识库")
    st.caption(f"已索引 {pipe.retriever.collection.count()} 个文本块")
    top_k = st.slider("引用块数 top-k", 2, 10, config.FINAL_TOP_K,
                      help="最终拼进 prompt 的文本块数量")
    use_reranker = st.toggle("cross-encoder 重排", value=True,
                             help="关掉后直接用混合检索的 RRF 排序,可对比效果")
    use_crag = st.toggle("CRAG 自纠错", value=False,
                         help="生成前让 LLM 过滤不相关的检索块,全不相关则改写查询重试。"
                              "会增加 API 调用和响应时间")
    use_check = st.toggle("生成后自检", value=False,
                          help="回答生成后让 LLM 核查是否有据可依,"
                               "不合格自动用更严格的指令重写一次")
    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button("重新加载索引", use_container_width=True,
                 help="重新运行过 python src/ingest.py 之后点这里,应用会读取新索引"):
        load_pipeline.clear()  # Discard handles to the collection replaced by ingest.
        st.rerun()
    st.divider()
    st.caption("流水线:混合检索(向量+BM25→RRF) → 重排 → DeepSeek 生成带引用回答")


if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(chunks: list[dict]):
    with st.expander(f"参考来源({len(chunks)} 块)"):
        for i, c in enumerate(chunks, 1):
            score = f" · 重排分 {c['rerank_score']:.3f}" if "rerank_score" in c else ""
            st.markdown(f"**[{i}] {c['source']} · 第{c['chunk_index']}块{score}**")
            st.caption(c["text"])


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])


if question := st.chat_input("输入你的问题…"):
    with st.chat_message("user"):
        st.markdown(question)

    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages]
    if history and history[-1]["role"] == "user":
        history.pop()  # Exclude an unanswered turn left by a failed request.
    # Persist the question even if generation fails.
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        try:
            with st.spinner("检索中…"):
                search_query, chunks, answer = pipe.ask(
                    question, history, top_k=top_k,
                    use_reranker=use_reranker, use_crag=use_crag,
                )
            if search_query != question:
                st.caption(f"检索查询已改写为:{search_query}")
            if pipeline_mod.low_confidence(chunks):
                st.warning("检索到的内容与问题相关性都很低,以下回答可能不可靠")
            full_answer = st.write_stream(answer)

            if use_check:
                verdict = llm.verify_answer(question, chunks, full_answer)
                if not verdict.get("supported", True):
                    st.caption(f"自检未通过({verdict.get('reason', '')}),已重新生成:")
                    full_answer = st.write_stream(
                        llm.answer_stream(question, chunks, history, strict=True)
                    )

            show_sources(chunks)
        except Exception as e:
            st.error(f"出错了:{e}")
            st.stop()

    st.session_state.messages.append(
        {"role": "assistant", "content": full_answer, "sources": chunks}
    )
