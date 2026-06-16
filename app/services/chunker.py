from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    start_char: int
    end_char: int


class TextChunker:
    """Split text into overlapping chunks while preferring natural boundaries."""

    def __init__(self, chunk_size: int = 1_000, chunk_overlap: int = 150) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._separators = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")

    def split(self, text: str) -> list[TextChunk]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []

        chunks: list[TextChunk] = []
        start = 0
        text_length = len(normalized)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            if end < text_length:
                end = self._best_boundary(normalized, start, end)

            chunk_text = normalized[start:end].strip()
            if chunk_text:
                chunks.append(
                    TextChunk(
                        index=len(chunks),
                        text=chunk_text,
                        start_char=start,
                        end_char=end,
                    )
                )

            if end >= text_length:
                break

            start = max(end - self.chunk_overlap, start + 1)

        return chunks

    def _best_boundary(self, text: str, start: int, hard_end: int) -> int:
        min_end = start + max(self.chunk_size // 2, 1)
        best = -1

        for separator in self._separators:
            candidate = text.rfind(separator, start, hard_end)
            if candidate >= min_end:
                best = max(best, candidate + len(separator))

        return best if best > start else hard_end
