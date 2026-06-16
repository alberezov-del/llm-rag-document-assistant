from app.services.chunker import TextChunker


def test_chunker_splits_long_text_with_overlap() -> None:
    text = " ".join(
        [f"Sentence {index} about retrieval augmented generation." for index in range(80)]
    )
    chunker = TextChunker(chunk_size=180, chunk_overlap=40)

    chunks = chunker.split(text)

    assert len(chunks) > 1
    assert all(chunk.text for chunk in chunks)
    assert all(len(chunk.text) <= 180 for chunk in chunks)
    assert chunks[1].start_char < chunks[0].end_char
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_chunker_returns_empty_list_for_blank_text() -> None:
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)

    assert chunker.split(" \n\n ") == []


def test_chunker_rejects_invalid_overlap() -> None:
    try:
        TextChunker(chunk_size=100, chunk_overlap=100)
    except ValueError as exc:
        assert "chunk_overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid overlap")
