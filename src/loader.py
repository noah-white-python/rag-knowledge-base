"""Load text, Markdown, and text-based PDF documents."""
from pathlib import Path

from pypdf import PdfReader

import config


def load_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gb18030")


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))

    return "\n".join(page.extract_text() or "" for page in reader.pages)


LOADERS = {
    ".txt": load_txt,
    ".md": load_txt,
    ".pdf": load_pdf,
}


def load_documents(data_dir: Path = config.DATA_DIR) -> list[dict]:
    """Load supported files recursively, using relative paths as source names."""
    docs = []
    for path in sorted(data_dir.rglob("*")):
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            continue
        text = loader(path).strip()
        if not text:
            print(f"[跳过] {path.name}:内容为空")
            continue
        # Relative paths keep IDs distinct for duplicate filenames in subdirectories.
        source = path.relative_to(data_dir).as_posix()
        docs.append({"source": source, "text": text})
        print(f"[加载] {source}:{len(text)} 字符")
    return docs


if __name__ == "__main__":
    documents = load_documents()
    print(f"\n共加载 {len(documents)} 个文档")
