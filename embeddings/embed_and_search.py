# embed_and_search.py
# Full pipeline: chunk docs → embed with nomic-embed-text → cosine search
# Run: python embed_and_search.py

import re
import json
import requests
import numpy as np

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL      = "nomic-embed-text"

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    cos(θ) = dot(A, B) / (|A| * |B|)
    Returns float in range [-1, 1].
    1.0  = identical direction (same meaning)
    0.0  = perpendicular     (unrelated)`
    -1.0 = opposite direction (opposite meaning)
    """
    dot    = np.dot(a, b)
    mag_a  = np.linalg.norm(a)
    mag_b  = np.linalg.norm(b)
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return float(dot / (mag_a * mag_b))


# ── 2. Embed one string via Ollama ────────────────────────────
def embed(text: str, prefix: str = "search_document") -> np.ndarray:
    """
    nomic-embed-text supports two prefixes:
      search_document: → for chunks being indexed
      search_query:    → for the user's question
    Using the right prefix noticeably improves search quality.
    """
    payload = f"{prefix}: {text}"
    resp = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": payload})
    resp.raise_for_status()
    return np.array(resp.json()["embedding"])   # shape: (768,)


def embed_query(text: str) -> np.ndarray:
    return embed(text, prefix="search_query")

def embed_document(text: str) -> np.ndarray:
    return embed(text, prefix="search_document")


# ── 3. Simple recursive chunker (from Topic 2) ────────────────
def chunk(text: str, size: int = 300, overlap: int = 40) -> list[str]:
    seps = ["\n\n", "\n", ". ", " "]

    def _split(t, seps):
        if len(t) <= size or not seps:
            return [t]
        sep = seps[0]
        if sep not in t:
            return _split(t, seps[1:])
        parts, buf, out = t.split(sep), "", []
        for p in parts:
            cand = buf + sep + p if buf else p
            if len(cand) <= size:
                buf = cand
            else:
                if buf: out.append(buf.strip())
                buf = _split(p, seps[1:])[0] if len(p) > size else p
        if buf.strip(): out.append(buf.strip())
        return out

    raw = _split(text, seps)
    return [(r + " " + raw[i+1][:overlap]).strip()
            if i+1 < len(raw) and overlap else r
            for i, r in enumerate(raw)]


# ── 4. Build a tiny in-memory index ──────────────────────────
class InMemoryIndex:
    """
    In production this would be ChromaDB / Qdrant / Pinecone.
    For now: a plain list of (chunk_text, vector) pairs.
    We'll replace this with ChromaDB in Topic 5 (Vector DBs).
    """
    def __init__(self):
        self.chunks: list[str]       = []
        self.vectors: list[np.ndarray] = []
        self.metadata: list[dict]    = []

    def add(self, text: str, vector: np.ndarray, meta: dict = None):
        self.chunks.append(text)
        self.vectors.append(vector)
        self.metadata.append(meta or {})

    def search(self, query_vec: np.ndarray, top_k: int = 3) -> list[dict]:
        if not self.vectors:
            return []
        scores = [cosine_similarity(query_vec, v) for v in self.vectors]
        ranked = sorted(zip(scores, self.chunks, self.metadata),
                        key=lambda x: x[0], reverse=True)
        return [
            {"score": round(s, 4), "text": t, "meta": m}
            for s, t, m in ranked[:top_k]
        ]


# ── 5. Index documents ────────────────────────────────────────
def index_document(
    index: InMemoryIndex,
    text: str,
    source: str,
    chunk_size: int = 300,
    overlap: int   = 40
) -> int:
    chunks = chunk(text, chunk_size, overlap)
    print(f"  Indexing '{source}' → {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        vec = embed_document(c)
        index.add(c, vec, meta={"source": source, "chunk_index": i})
        print(f"    [{i+1}/{len(chunks)}] {len(c)} chars → vec({len(vec)} dims)")
    return len(chunks)


# ── 6. Search and display results ────────────────────────────
def search(index: InMemoryIndex, query: str, top_k: int = 3):
    print(f"\n{'─'*55}")
    print(f"  Query: {query}")
    print(f"{'─'*55}")
    query_vec = embed_query(query)
    results   = index.search(query_vec, top_k)

    for rank, r in enumerate(results, 1):
        score = r['score']
        bar   = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"\n  #{rank}  score={score:.4f}  {bar}")
        print(f"  source: {r['meta'].get('source')}  chunk: {r['meta'].get('chunk_index')}")
        print(f"  text:   {r['text'][:120]}...")
    return results


# ── 7. Run it ─────────────────────────────────────────────────
if __name__ == "__main__":

    # Your SaaS support docs from Topic 2
    DOCS = {
        "changelog_v2.4.md": """
v2.4.0 released May 2026.
New features: Smart filters let users filter tasks by assignee, due date, label, or custom field.
Bulk edit lets you select multiple tasks and update status or priority at once.
Full keyboard navigation added. Press question mark to see all shortcuts.

Bug fixes: Fixed notifications not clearing after being read on mobile.
Resolved date picker showing wrong month with negative timezone offsets.
Fixed duplicate webhook events on task creation.

Performance: Dashboard load time reduced by 40 percent through improved query batching.
Real-time sync latency dropped from 800ms to 120ms average.
        """,

        "api_reference.md": """
Authentication: All API requests require a Bearer token in the Authorization header.
Tokens are created in Settings under API Keys. Tokens can be scoped to read-only or read-write.

Create a task: POST /api/v1/tasks
Required fields: title (string, max 256 chars) and project_id (string).
Optional fields: assignee_id, due_date in ISO 8601, priority as low medium high or urgent.

List tasks: GET /api/v1/tasks
Returns paginated results. Filter by project_id, status, assignee.
Status values: open, in_progress, done, cancelled.
Default limit is 25, maximum is 100. Use cursor for pagination.
        """,

        "feature_guide.md": """
Projects are the top-level containers for your work.
Each project has its own task list, members, and settings.

Creating a project: Click New Project in the left sidebar.
Enter a name and optional description. Choose a color and icon.
Set visibility to private for invite-only or workspace for all members.

Inviting members: Open project Settings then the Members tab.
Enter email and choose a role.
Viewer can see tasks but not edit. Member can create and edit tasks.
Admin has full access including settings and member management.
Invites expire after 7 days.

Workflows define the status columns in your project.
Default statuses are Backlog, In Progress, In Review, and Done.
Customise in Settings under Workflow.
        """,
    }

    print("\nBuilding index...")
    print("=" * 55)
    idx = InMemoryIndex()
    for filename, content in DOCS.items():
        index_document(idx, content.strip(), source=filename)

    total = len(idx.chunks)
    dims  = len(idx.vectors[0]) if idx.vectors else 0
    print(f"\nIndex ready: {total} chunks × {dims} dims each")

    # ── Test queries ──────────────────────────────────────────
    queries = [
        "How do I invite someone to my project?",
        "What changed in the latest release?",
        "How do I create a task using the API?",
        "Why is my dashboard loading slowly?",
        "What does the admin role allow?",
    ]

    print("\n\nRunning search queries...")
    for q in queries:
        results = search(idx, q, top_k=2)

    # ── Save index to disk (so you don't re-embed every run) ──
    print("\n\nSaving index to disk...")
    save_data = {
        "chunks":   idx.chunks,
        "vectors":  [v.tolist() for v in idx.vectors],
        "metadata": idx.metadata,
    }
    with open("index.json", "w") as f:
        json.dump(save_data, f)
    print("  Saved → index.json")
    print(f"  Size  → {len(json.dumps(save_data)) // 1024} KB")


# ── Bonus: load the saved index and query it ──────────────────
def load_and_query(index_path: str, query: str):
    """Load a saved index and search it — no re-embedding needed."""
    with open(index_path) as f:
        data = json.load(f)

    idx = InMemoryIndex()
    for text, vec, meta in zip(data["chunks"], data["vectors"], data["metadata"]):
        idx.add(text, np.array(vec), meta)

    print(f"Loaded {len(idx.chunks)} chunks from {index_path}")
    return search(idx, query)