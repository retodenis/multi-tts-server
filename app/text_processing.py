from __future__ import annotations

import re

# Tags commonly appended by LLM-driven clients (SkyrimNet) that should not be
# spoken aloud: [thinking], <STAGE_1>, (whispers), etc.
_TAG_RE = re.compile(r"<[^>]+>|\[[^\]]{0,40}\]")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def strip_tags(text: str) -> str:
    cleaned = _TAG_RE.sub(" ", text)
    return _MULTI_SPACE_RE.sub(" ", cleaned).strip()


def clean_text(text: str, strip: bool = True) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    if strip:
        text = strip_tags(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str, max_len: int = 500) -> list[str]:
    """Split text into sentence-length chunks (RU/EN punctuation aware)."""
    text = clean_text(text, strip=False)
    if not text:
        return []

    parts: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for chunk in _split_on_punct(line):
            if len(chunk) > max_len:
                parts.extend(_hard_split(chunk, max_len))
            else:
                parts.append(chunk)
    return [p for p in parts if p]


def _split_on_punct(text: str) -> list[str]:
    tokens = re.split(r"(?<=[.!?…!?»\"])[\s]+", text)
    return [t.strip() for t in tokens if t.strip()]


def _hard_split(text: str, max_len: int) -> list[str]:
    words = text.split()
    chunks, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_len:
            chunks.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        chunks.append(cur)
    return chunks
