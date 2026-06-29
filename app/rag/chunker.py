from dataclasses import dataclass
import re


DEFAULT_TARGET_CHUNK_CHARS = 1200
DEFAULT_MAX_CHUNK_CHARS = 2000
PREVIEW_CHARS = 200


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    char_count: int


def chunk_text(
    text: str,
    target_chunk_chars: int = DEFAULT_TARGET_CHUNK_CHARS,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[TextChunk]:
    if not text.strip():
        raise ValueError("text must be non-empty")

    if target_chunk_chars <= 0:
        raise ValueError("target_chunk_chars must be positive")

    if max_chunk_chars < target_chunk_chars:
        raise ValueError(
            "max_chunk_chars must be greater than or equal to target_chunk_chars"
        )

    paragraphs = _split_paragraphs(text)
    chunk_texts: list[str] = []
    current_paragraphs: list[str] = []
    current_char_count = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chunk_chars:
            _flush_current_chunk(current_paragraphs, chunk_texts)
            current_char_count = 0
            chunk_texts.extend(
                _split_large_paragraph(
                    paragraph=paragraph,
                    max_chunk_chars=max_chunk_chars,
                )
            )
            continue

        separator_chars = 2 if current_paragraphs else 0
        next_char_count = current_char_count + separator_chars + len(paragraph)

        if current_paragraphs and next_char_count > max_chunk_chars:
            _flush_current_chunk(current_paragraphs, chunk_texts)
            current_paragraphs = [paragraph]
            current_char_count = len(paragraph)
            if current_char_count >= target_chunk_chars:
                _flush_current_chunk(current_paragraphs, chunk_texts)
                current_char_count = 0
            continue

        current_paragraphs.append(paragraph)
        current_char_count = next_char_count

        if current_char_count >= target_chunk_chars:
            _flush_current_chunk(current_paragraphs, chunk_texts)
            current_char_count = 0

    _flush_current_chunk(current_paragraphs, chunk_texts)

    return [
        TextChunk(index=index, text=chunk, char_count=len(chunk))
        for index, chunk in enumerate(chunk_texts)
    ]


def _split_paragraphs(text: str) -> list[str]:
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n+", normalized_text)
        if paragraph.strip()
    ]


def _flush_current_chunk(
    current_paragraphs: list[str],
    chunk_texts: list[str],
) -> None:
    if not current_paragraphs:
        return

    chunk_texts.append("\n\n".join(current_paragraphs))
    current_paragraphs.clear()


def _split_large_paragraph(
    paragraph: str,
    max_chunk_chars: int,
) -> list[str]:
    pieces: list[str] = []
    remaining_text = paragraph.strip()

    while len(remaining_text) > max_chunk_chars:
        split_at = _find_safe_split(remaining_text, max_chunk_chars)
        piece = remaining_text[:split_at].strip()
        if piece:
            pieces.append(piece)
        remaining_text = remaining_text[split_at:].strip()

    if remaining_text:
        pieces.append(remaining_text)

    return pieces


def _find_safe_split(text: str, max_chunk_chars: int) -> int:
    split_window = text[: max_chunk_chars + 1]
    minimum_useful_split = max_chunk_chars // 2

    for boundary in (". ", "! ", "? ", "; ", ": ", ", ", " "):
        split_at = split_window.rfind(boundary)
        if split_at >= minimum_useful_split:
            return split_at + len(boundary.rstrip())

    return max_chunk_chars
