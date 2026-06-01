# doc_chunker_pipeline.py
# Production chunking pipeline for a SaaS support bot
# Handles: changelog, API docs, feature guides

import re
import json
import hashlib
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional
import requests

# ── Config ────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

class DocType(Enum):
    CHANGELOG      = "changelog"
    API_DOCS       = "api_docs"
    FEATURE_GUIDE  = "feature_guide"
    UNKNOWN        = "unknown"


# ── Data model (what goes into your vector DB later) ──────────
@dataclass
class EmbedReadyChunk:
    chunk_id:       str          # stable hash-based ID
    doc_type:       str          # changelog / api_docs / feature_guide
    source_file:    str          # e.g. "changelog_v2.4.md"
    chunk_index:    int
    char_count:     int
    token_estimate: int          # rough: chars / 4
    text:           str          # clean chunk text
    embed_input:    str          # what actually goes to nomic-embed-text
    embedding:      Optional[list[float]] = None


# ── Stage 1: Detect doc type ──────────────────────────────────
def detect_doc_type(filename: str, text: str) -> DocType:
    fn = filename.lower()
    if "changelog" in fn or "release" in fn:    return DocType.CHANGELOG
    if "api" in fn or "reference" in fn:        return DocType.API_DOCS
    if "guide" in fn or "tutorial" in fn:       return DocType.FEATURE_GUIDE
    # Fallback: check content signals
    if re.search(r"##\s*v\d+\.\d+", text):     return DocType.CHANGELOG
    if re.search(r"(GET|POST|PUT|DELETE)\s+/", text): return DocType.API_DOCS
    return DocType.UNKNOWN


# ── Stage 2: Clean markdown ───────────────────────────────────
def clean_markdown(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", "[code block]", text)  # strip code blocks
    text = re.sub(r"`([^`]+)`", r"\1", text)                # inline code → plain
    text = re.sub(r"#{1,6}\s+", "", text)                   # remove headings #
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)            # **bold** → plain
    text = re.sub(r"\*(.*?)\*", r"\1", text)                # *italic* → plain
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)         # [link](url) → text
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)    # bullet points
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.M)    # numbered lists
    text = re.sub(r"\n{3,}", "\n\n", text)                  # collapse blank lines
    return text.strip()


# ── Stage 3: Recursive chunker ────────────────────────────────
def recursive_chunk(text: str, chunk_size: int = 300, overlap: int = 40) -> list[str]:
    separators = ["\n\n", "\n", ". ", " "]

    def _split(t: str, seps: list[str]) -> list[str]:
        if len(t) <= chunk_size or not seps:
            return [t]
        sep = seps[0]
        if sep not in t:
            return _split(t, seps[1:])
        parts = t.split(sep)
        results, buf = [], ""
        for part in parts:
            candidate = buf + sep + part if buf else part
            if len(candidate) <= chunk_size:
                buf = candidate
            else:
                if buf: results.append(buf.strip())
                buf = _split(part, seps[1:])[0] if len(part) > chunk_size else part
        if buf.strip(): results.append(buf.strip())
        return results

    raw = _split(text, separators)

    # Add overlap from next chunk
    final = []
    for i, chunk in enumerate(raw):
        if i + 1 < len(raw) and overlap > 0:
            chunk = chunk + " " + raw[i + 1][:overlap]
        final.append(chunk.strip())
    return final


# ── Stage 4: Enrich with metadata ────────────────────────────
def make_chunk_id(source: str, index: int, text: str) -> str:
    """Stable deterministic ID — same content always gets same ID."""
    payload = f"{source}::{index}::{text[:50]}"
    return hashlib.md5(payload.encode()).hexdigest()[:12]

def build_embed_input(chunk: str, doc_type: DocType) -> str:
    """
    nomic-embed-text works best with the search_document prefix.
    We also inject doc_type so the model has context about what kind of text this is.
    """
    label = doc_type.value.replace("_", " ")
    return f"search_document: [{label}] {chunk}"

def enrich_chunks(
    chunks: list[str],
    doc_type: DocType,
    source_file: str
) -> list[EmbedReadyChunk]:
    result = []
    for i, text in enumerate(chunks):
        result.append(EmbedReadyChunk(
            chunk_id       = make_chunk_id(source_file, i, text),
            doc_type       = doc_type.value,
            source_file    = source_file,
            chunk_index    = i,
            char_count     = len(text),
            token_estimate = len(text) // 4,
            text           = text,
            embed_input    = build_embed_input(text, doc_type),
        ))
    return result


# ── Stage 5: Embed ────────────────────────────────────────────
def embed_chunk(chunk: EmbedReadyChunk) -> EmbedReadyChunk:
    """Call Ollama and attach the vector to the chunk."""
    resp = requests.post(OLLAMA_URL, json={
        "model": EMBED_MODEL,
        "prompt": chunk.embed_input
    })
    resp.raise_for_status()
    chunk.embedding = resp.json()["embedding"]
    return chunk


# ── Full pipeline ─────────────────────────────────────────────
def process_document(
    text: str,
    filename: str,
    chunk_size: int = 300,
    overlap: int = 40,
    embed: bool = True
) -> list[EmbedReadyChunk]:

    print(f"\n{'='*55}")
    print(f"  Processing: {filename}")
    print(f"{'='*55}")

    # Stage 1: detect
    doc_type = detect_doc_type(filename, text)
    print(f"  Doc type    : {doc_type.value}")
    print(f"  Raw chars   : {len(text)}")

    # Stage 2: clean
    cleaned = clean_markdown(text)
    print(f"  Clean chars : {len(cleaned)}  (removed {len(text)-len(cleaned)} chars of markdown)")

    # Stage 3: chunk
    chunks = recursive_chunk(cleaned, chunk_size, overlap)
    print(f"  Chunks      : {len(chunks)}  (avg {sum(len(c) for c in chunks)//len(chunks)} chars)")

    # Stage 4: enrich
    enriched = enrich_chunks(chunks, doc_type, filename)

    # Stage 5: embed (optional — skip if testing chunking only)
    if embed:
        print(f"  Embedding {len(enriched)} chunks with {EMBED_MODEL}...")
        embedded = []
        for c in enriched:
            ec = embed_chunk(c)
            embedded.append(ec)
            print(f"    [{c.chunk_index+1}/{len(enriched)}] chunk_id={c.chunk_id}  vec_dims={len(ec.embedding)}")
        return embedded

    return enriched


# ── Test it ───────────────────────────────────────────────────
if __name__ == "__main__":
    CHANGELOG = """## v2.4.0 — May 2026

### New features
- **Smart filters**: Filter tasks by assignee, due date, label, or custom field.
- **Bulk edit**: Select multiple tasks and update status or priority at once.
- **Keyboard shortcuts**: Full keyboard navigation. Press ? to see all shortcuts.

### Bug fixes
- Fixed notifications not clearing after being read on mobile.
- Resolved date picker showing wrong month with negative timezone offset.

### Performance
- Dashboard load time reduced by 40% through improved query batching.
- Real-time sync latency dropped from 800ms to 120ms average."""

    API_DOCS = """# Authentication

All API requests need a Bearer token in the Authorization header.

Tokens can be created in Settings > API Keys. Each token is scoped to read-only or read-write.

# Tasks API

## Create a task

POST /api/v1/tasks

Creates a new task in the specified project. project_id and title are required.

- title (string, required): Task title, max 256 characters.
- project_id (string, required): Project this task belongs to.
- assignee_id (string, optional): User ID of the assignee.
- priority (enum, optional): low | medium | high | urgent."""

    # Process without embedding first (faster for testing)
    changelog_chunks = process_document(CHANGELOG, "changelog_v2.4.md", embed=False)
    api_chunks = process_document(API_DOCS, "api_reference.md", embed=False)

    print("\n\nSample chunk output (as JSON):")
    sample = changelog_chunks[0]
    sample_dict = asdict(sample)
    sample_dict.pop("embedding")  # None when not embedded
    print(json.dumps(sample_dict, indent=2))

    # Now embed one doc (set embed=True when Ollama is running)
    # chunks_with_vectors = process_document(CHANGELOG, "changelog_v2.4.md", embed=True)
    # print(f"\nFirst chunk vector: {chunks_with_vectors[0].embedding[:5]}...")