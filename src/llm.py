"""DeepSeek calls for query rewriting, relevance checks, and answer generation."""
import json

from openai import OpenAI

import config

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.DEEPSEEK_API_KEY or config.DEEPSEEK_API_KEY == "sk-your-key-here":
            raise RuntimeError(
                "没有配置 DEEPSEEK_API_KEY:请编辑项目根目录的 .env 文件,填入你的 key"
            )
        _client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.LLM_BASE_URL)
    return _client


def chat(messages: list[dict], temperature: float = config.LLM_TEMPERATURE,
         stream: bool = False, json_mode: bool = False):
    """Send a chat completion request, optionally using streaming or JSON output."""
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return get_client().chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=temperature,
        stream=stream,
        extra_body={"thinking": {"type": "disabled"}},
        **kwargs,
    )


CONDENSE_PROMPT = """\
你的任务是改写用户的追问。下面是对话历史和用户的最新问题,
最新问题可能包含"它/这个/为什么呢"之类依赖上文的指代,
请把它改写成一个不看历史也能理解的独立问题,用于检索知识库。
只输出改写后的问题本身,不要任何解释。

对话历史:
{history}

最新问题:{question}
改写后:"""


def condense_question(history: list[dict], question: str) -> str:
    """Resolve references in a follow-up using recent conversation history."""
    if not history:
        return question
    recent = history[-2 * config.MAX_HISTORY_TURNS:]
    history_text = "\n".join(
        f"{'用户' if m['role'] == 'user' else '助手'}:{m['content']}" for m in recent
    )
    response = chat(
        [{"role": "user",
          "content": CONDENSE_PROMPT.format(history=history_text, question=question)}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


GRADE_PROMPT = """\
下面是一个问题和 {n} 个检索到的文档块。逐个判断每块是否与问题相关
(能为回答这个问题提供有用信息才算相关)。
以 JSON 输出:{{"relevant": [true, false, …]}},数组长度必须正好是 {n}。

问题:{question}

{blocks}"""


def grade_chunks(question: str, chunks: list[dict]) -> list[bool]:
    """Classify chunk relevance in a single JSON-mode request."""
    blocks = "\n\n".join(f"[块{i}]\n{c['text']}" for i, c in enumerate(chunks, 1))
    response = chat(
        [{"role": "user",
          "content": GRADE_PROMPT.format(n=len(chunks), question=question, blocks=blocks)}],
        temperature=0,
        json_mode=True,
    )
    flags = json.loads(response.choices[0].message.content).get("relevant", [])
    if len(flags) != len(chunks):
        return [True] * len(chunks)  # Keep all chunks if the relevance response has the wrong length.
    return [bool(f) for f in flags]


REWRITE_PROMPT = """\
下面这个问题在知识库里没有检索到相关内容,可能是表述不适合检索。
请把它改写成更适合检索的形式:补全省略的信息,换用更规范的术语,保持原意。
只输出改写后的问题本身。

问题:{question}"""


def rewrite_query(question: str) -> str:
    """Rewrite a question for a second retrieval attempt."""
    response = chat(
        [{"role": "user", "content": REWRITE_PROMPT.format(question=question)}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def build_context(chunks: list[dict]) -> str:
    """Format numbered chunks and their sources for citation."""
    return "\n\n".join(
        f"[{i}] 来源:{c['source']}\n{c['text']}" for i, c in enumerate(chunks, 1)
    )


SYSTEM_PROMPT = """\
你是一个严谨的知识库问答助手。请严格依据下面提供的上下文回答用户问题,遵守:
1. 只使用上下文中的信息,上下文中没有的内容,明确回答"知识库中没有找到相关信息",绝不编造;
2. 回答中的每个论断都要标注来源编号,格式如 [1]、[2],可以一句话引用多个来源;
3. 用中文回答,条理清晰,直接回答问题本身。

上下文:
{context}"""


def answer_stream(question: str, chunks: list[dict], history: list[dict] | None = None,
                  strict: bool = False):
    """Yield answer text; strict mode requests an additional grounding check."""
    system = SYSTEM_PROMPT.format(context=build_context(chunks))
    if strict:
        system += ("\n\n注意:上一次生成的回答自检未通过(包含上下文不支持的内容)。"
                   "这次请逐句核对上下文,没有依据的句子一律删掉,"
                   "或改为明确说明知识库中没有该信息。")
    messages = [{"role": "system", "content": system}]
    if history:
        messages += history[-2 * config.MAX_HISTORY_TURNS:]
    messages.append({"role": "user", "content": question})

    for event in chat(messages, stream=True):
        delta = event.choices[0].delta.content
        if delta:
            yield delta


VERIFY_PROMPT = """\
你是核查员。判断下面这个回答的每个论断是否都能在上下文中找到依据。
回答"知识库中没有找到相关信息"这类如实拒答也算合格。
只输出 JSON,格式:{{"supported": true 或 false, "reason": "一句话说明"}}

[问题]
{question}

[上下文]
{context}

[回答]
{answer}"""


def verify_answer(question: str, chunks: list[dict], answer: str) -> dict:
    """Ask the model whether the answer is supported by the retrieved context."""
    response = chat(
        [{"role": "user", "content": VERIFY_PROMPT.format(
            question=question, context=build_context(chunks), answer=answer)}],
        temperature=0,
        json_mode=True,
    )
    return json.loads(response.choices[0].message.content)
