"""Split documents into overlapping chunks with source and section metadata."""
import re

import config


SEPARATORS = ["\n\n", "\n", "。", "!", "?", ". ", " "]


def _split_text(text: str, separators: list[str]) -> list[str]:
    """Split oversized text at progressively finer separators."""
    if len(text) <= config.CHUNK_SIZE:
        return [text]
    if not separators:
        return [text[i:i + config.CHUNK_SIZE] for i in range(0, len(text), config.CHUNK_SIZE)]

    sep, rest = separators[0], separators[1:]

    parts = [p + sep for p in text.split(sep)]
    parts[-1] = parts[-1][:-len(sep)] if parts[-1].endswith(sep) else parts[-1]

    pieces = []
    for part in parts:
        if len(part) > config.CHUNK_SIZE:
            pieces.extend(_split_text(part, rest))
        elif part.strip():
            pieces.append(part)
    return pieces


def _merge_pieces(pieces: list[str]) -> list[str]:
    """Merge pieces up to CHUNK_SIZE, retaining overlap where space allows."""
    chunks = []
    current = ""
    for piece in pieces:
        if len(current) + len(piece) > config.CHUNK_SIZE and current:
            chunks.append(current)

            overlap = min(config.CHUNK_OVERLAP, config.CHUNK_SIZE - len(piece))
            current = (current[-overlap:] if overlap > 0 else "") + piece
        else:
            current += piece
    if current.strip():
        chunks.append(current)
    return chunks


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Return heading paths and section bodies, ignoring headings inside code fences."""
    sections = []
    titles: dict[int, str] = {}
    body: list[str] = []

    def flush():
        content = "\n".join(body).strip()
        if content:
            path = " · ".join(titles[level] for level in sorted(titles))
            sections.append((path, content))
        body.clear()

    in_code_block = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_code_block = not in_code_block
            body.append(line)
            continue
        heading = None if in_code_block else re.match(r"^(#{1,6})\s+(.+)", line)
        if heading:
            flush()
            level = len(heading.group(1))
            titles = {lv: t for lv, t in titles.items() if lv < level}
            titles[level] = heading.group(2).strip()
        else:
            body.append(line)
    flush()
    return sections


def chunk_document(doc: dict) -> list[dict]:
    """Return chunks with stable source-relative IDs and heading prefixes."""
    text = re.sub(r"\n{3,}", "\n\n", doc["text"])
    if doc["source"].lower().endswith(".md"):
        sections = _split_markdown_sections(text)
    else:
        sections = [("", text)]

    texts = []
    for title, body in sections:
        prefix = f"【{doc['source']}" + (f" · {title}" if title else "") + "】\n"
        for piece in _merge_pieces(_split_text(body, SEPARATORS)):
            texts.append(prefix + piece.strip())

    return [
        {
            "id": f"{doc['source']}::chunk_{i}",
            "source": doc["source"],
            "chunk_index": i,
            "text": chunk,
        }
        for i, chunk in enumerate(texts)
    ]


def chunk_documents(docs: list[dict]) -> list[dict]:
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc)
        print(f"[切块] {doc['source']}:{len(chunks)} 块")
        all_chunks.extend(chunks)
    return all_chunks
