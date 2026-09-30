import re
from typing import Dict, List, Optional


def create_chunks(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
    metadata: Optional[Dict] = None
) -> List[Dict]:
    """
    Splits input text into word-based chunks with configurable size and overlap.

    Parameters:
    - text: The clean source text to chunk.
    - chunk_size: Number of words per chunk (default: 400). Must be > 0.
    - overlap: Number of overlapping words between consecutive chunks (default: 50). Must be >= 0 and < chunk_size.
    - metadata: Optional dictionary of source metadata (e.g. source_name, source_type).

    Returns:
    List of chunk dictionaries with metadata:
    [
        {
            "chunk_id": 0,
            "source_name": "...",
            "source_type": "...",
            "text": "...",
            "start_word": 0,
            "end_word": 400,
            "word_count": 400
        },
        ...
    ]
    """
    # Parameter validations
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})")

    if not text or not text.strip():
        return []

    # Split text into words based on whitespace
    words = text.strip().split()
    total_words = len(words)
    if total_words == 0:
        return []

    step = chunk_size - overlap
    chunks = []
    chunk_idx = 0
    start = 0

    base_meta = metadata or {}
    source_name = base_meta.get("source_name", "unknown_source")
    source_type = base_meta.get("source_type", "text")

    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()

        if chunk_text:
            chunk_item = {
                "chunk_id": chunk_idx,
                "source_name": source_name,
                "source_type": source_type,
                "text": chunk_text,
                "start_word": start,
                "end_word": end,
                "word_count": len(chunk_words)
            }
            chunks.append(chunk_item)
            chunk_idx += 1

        # If we reached or exceeded the end of the text, break
        if end >= total_words:
            break

        start += step

    return chunks


def chunk_sources(
    sources: List[Dict],
    chunk_size: int = 400,
    overlap: int = 50
) -> List[Dict]:
    """
    Processes a list of structured source dictionaries (from document ingestion)
    and returns a combined list of chunks with globally unique, sequential chunk_ids.

    Parameters:
    - sources: List of dicts with keys 'source_name', 'source_type', 'content'.
    - chunk_size: Word count per chunk.
    - overlap: Overlapping word count.

    Returns:
    Combined list of chunks across all sources with unique sequential chunk_ids.
    """
    all_chunks = []
    global_chunk_id = 0

    for src in sources:
        content = src.get("content", "")
        if not content:
            continue

        meta = {
            "source_name": src.get("source_name", "unknown"),
            "source_type": src.get("source_type", "document")
        }

        source_chunks = create_chunks(
            text=content,
            chunk_size=chunk_size,
            overlap=overlap,
            metadata=meta
        )

        for chunk in source_chunks:
            chunk["chunk_id"] = global_chunk_id
            all_chunks.append(chunk)
            global_chunk_id += 1

    return all_chunks
