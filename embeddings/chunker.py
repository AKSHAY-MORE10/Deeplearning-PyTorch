# chunker.py  — all 4 strategies, production-grade

import re
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    index: int
    start_char: int
    end_char: int
    strategy: str

# ── 1. Fixed size ──────────────────────────────────────────────
def chunk_fixed(text: str, chunk_size: int = 200, overlap: int = 40) -> list[Chunk]:
    """Split every N chars with overlap. Fast but dumb — ignores word/sentence boundaries."""
    chunks = []
    i = 0
    idx = 0
    while i < len(text):
        end = min(i + chunk_size, len(text))
        chunks.append(Chunk(
            text=text[i:end],
            index=idx,
            start_char=i,
            end_char=end,
            strategy="fixed"
        ))
        i += chunk_size - overlap
        idx += 1
    return chunks


# ── 2. Sentence-aware ─────────────────────────────────────────
def chunk_sentences(text: str, max_chunk_size: int = 300, overlap_sentences: int = 1) -> list[Chunk]:
    """Group sentences together up to max_chunk_size. Overlaps by N sentences."""
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    buf_sentences = []
    buf_len = 0
    start_char = 0
    idx = 0

    for sent in sentences:
        if buf_len + len(sent) > max_chunk_size and buf_sentences:
            chunk_text = ' '.join(buf_sentences)
            chunks.append(Chunk(
                text=chunk_text,
                index=idx,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                strategy="sentence"
            ))
            idx += 1
            # Keep last N sentences as overlap for next chunk
            buf_sentences = buf_sentences[-overlap_sentences:] if overlap_sentences else []
            buf_len = sum(len(s) for s in buf_sentences)
            start_char = text.find(buf_sentences[0]) if buf_sentences else start_char + len(chunk_text)

        buf_sentences.append(sent)
        buf_len += len(sent)

    if buf_sentences:
        chunk_text = ' '.join(buf_sentences)
        chunks.append(Chunk(
            text=chunk_text,
            index=idx,
            start_char=start_char,
            end_char=start_char + len(chunk_text),
            strategy="sentence"
        ))

    return chunks


# ── 3. Paragraph ──────────────────────────────────────────────
def chunk_paragraphs(text: str, max_chunk_size: int = 500) -> list[Chunk]:
    """Split on blank lines. If a paragraph is too long, fall back to sentences."""
    paragraphs = re.split(r'\n\s*\n', text.strip())
    chunks = []
    pos = 0
    idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chunk_size:
            # Paragraph too big — break it by sentences
            sub_chunks = chunk_sentences(para, max_chunk_size)
            for sc in sub_chunks:
                chunks.append(Chunk(
                    text=sc.text,
                    index=idx,
                    start_char=pos,
                    end_char=pos + len(sc.text),
                    strategy="paragraph"
                ))
                idx += 1
        else:
            chunks.append(Chunk(
                text=para,
                index=idx,
                start_char=pos,
                end_char=pos + len(para),
                strategy="paragraph"
            ))
            idx += 1
        pos += len(para) + 2  # +2 for \n\n

    return chunks


# ── 4. Recursive (production default) ────────────────────────
def chunk_recursive(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
    separators: list[str] = None
) -> list[Chunk]:
    """
    LangChain-style recursive splitter.
    Tries each separator in order until chunks fit within chunk_size.
    This is what most production RAG pipelines use.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, seps: list[str]) -> list[str]:
        if not seps:
            return [text]
        sep = seps[0]
        if sep == "":
            # Base case — split every character
            return list(text)

        splits = text.split(sep)
        result = []
        buf = ""

        for piece in splits:
            candidate = buf + sep + piece if buf else piece
            if len(candidate) <= chunk_size:
                buf = candidate
            else:
                if buf:
                    result.append(buf)
                # Piece itself too big — go deeper
                if len(piece) > chunk_size:
                    result.extend(_split(piece, seps[1:]))
                    buf = ""
                else:
                    buf = piece

        if buf:
            result.append(buf)
        return result

    raw_chunks = _split(text, separators)

    # Add overlap by appending start of next chunk to current
    chunks = []
    for i, raw in enumerate(raw_chunks):
        # Build overlap suffix from the next chunk
        overlap_text = ""
        if i + 1 < len(raw_chunks) and overlap > 0:
            overlap_text = " " + raw_chunks[i + 1][:overlap]
        final_text = (raw + overlap_text).strip()
        chunks.append(Chunk(
            text=final_text,
            index=i,
            start_char=0,  # simplified — tracking exact positions needs more work
            end_char=len(final_text),
            strategy="recursive"
        ))

    return chunks


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """Machine learning is a subset of artificial intelligence.
It enables systems to learn from data automatically.

Neural networks are inspired by the human brain structure.
Deep learning uses many layers of neurons.

Transformers revolutionised natural language processing.
Attention mechanisms allow models to focus on relevant tokens.
Embeddings convert text into dense vector representations."""

    print("=" * 60)
    print("RECURSIVE CHUNKING (production default)")
    print("=" * 60)
    chunks = chunk_recursive(sample, chunk_size=150, overlap=30)
    for c in chunks:
        print(f"\n[Chunk {c.index+1}] {len(c.text)} chars")
        print(f"  {c.text[:80]}{'...' if len(c.text)>80 else ''}")

    print("\n" + "=" * 60)
    print("SENTENCE CHUNKING")
    print("=" * 60)
    chunks = chunk_sentences(sample, max_chunk_size=150)
    for c in chunks:
        print(f"\n[Chunk {c.index+1}] {len(c.text)} chars")
        print(f"  {c.text[:80]}{'...' if len(c.text)>80 else ''}")