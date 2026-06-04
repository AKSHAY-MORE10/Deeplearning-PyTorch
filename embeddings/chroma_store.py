# chroma_store.py
# Drop-in replacement for the InMemoryIndex from Topic 3
# Everything persists to ./chroma_db/ on disk automatically

import chromadb
import requests
import numpy as np
from chromadb.config import Settings

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL      = "nomic-embed-text"


# ── Embed helpers (same as Topic 3) ───────────────────────────
def embed(text: str, prefix: str = "search_document") -> list[float]:
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": f"{prefix}: {text}"
    })
    resp.raise_for_status()
    return resp.json()["embedding"]   # plain list[float] — chroma wants this

def embed_query(text: str)    -> list[float]: return embed(text, "search_query")
def embed_document(text: str) -> list[float]: return embed(text, "search_document")


# ── ChromaDB client ───────────────────────────────────────────
def get_client(path: str = "./chroma_db") -> chromadb.PersistentClient:
    """
    PersistentClient writes to disk at `path`.
    Data survives process restarts — no re-embedding needed.
    """
    return chromadb.PersistentClient(path=path)


def get_collection(client: chromadb.PersistentClient, name: str):
    """
    get_or_create_collection:
    - First run  → creates the collection fresh
    - Later runs → loads the existing one from disk
    """
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}  # use cosine distance (matches our Topic 3 math)
    )


# ── Index documents ────────────────────────────────────────────
def index_document(
    collection,
    text: str,
    source: str,
    chunk_size: int = 300,
    overlap: int   = 40
):
    """Chunk → embed → upsert into ChromaDB collection."""
    chunks = recursive_chunk(text, chunk_size, overlap)
    print(f"\n  Indexing '{source}' → {len(chunks)} chunks")

    ids        = []
    embeddings = []
    documents  = []
    metadatas  = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{source}__chunk_{i}"
        vec      = embed_document(chunk)

        ids.append(chunk_id)
        embeddings.append(vec)
        documents.append(chunk)
        metadatas.append({
            "source":      source,
            "chunk_index": i,
            "char_count":  len(chunk),
        })
        print(f"    [{i+1}/{len(chunks)}] id={chunk_id}  dims={len(vec)}")

    # upsert = insert if new, update if already exists (safe to re-run)
    collection.upsert(
        ids        = ids,
        embeddings = embeddings,
        documents  = documents,
        metadatas  = metadatas,
    )
    print(f"  Upserted {len(chunks)} chunks into collection")


# ── Search ────────────────────────────────────────────────────
def search(collection, query: str, top_k: int = 3) -> list[dict]:
    """Embed query → HNSW search → return ranked results."""
    query_vec = embed_query(query)

    results = collection.query(
        query_embeddings = [query_vec],
        n_results        = top_k,
        include          = ["documents", "metadatas", "distances"]
    )

    # ChromaDB returns distances (lower = more similar for cosine)
    # Convert to similarity score: score = 1 - distance
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    output = []
    for doc, meta, dist in zip(docs, metas, distances):
        output.append({
            "score":  round(1 - dist, 4),
            "text":   doc,
            "source": meta.get("source"),
            "chunk":  meta.get("chunk_index"),
        })
    return output


def print_results(query: str, results: list[dict]):
    print(f"\n{'─'*55}")
    print(f"  Query: {query}")
    print(f"{'─'*55}")
    for i, r in enumerate(results, 1):
        bar = "█" * int(r["score"] * 20) + "░" * (20 - int(r["score"] * 20))
        print(f"\n  #{i}  score={r['score']:.4f}  {bar}")
        print(f"  {r['source']}  chunk #{r['chunk']}")
        print(f"  {r['text'][:110]}...")


# ── Recursive chunker (same as before) ───────────────────────
def recursive_chunk(text: str, size: int = 300, overlap: int = 40) -> list[str]:
    seps = ["\n\n", "\n", ". ", " "]
    def _split(t, seps):
        if len(t) <= size or not seps: return [t]
        sep = seps[0]
        if sep not in t: return _split(t, seps[1:])
        parts, buf, out = t.split(sep), "", []
        for p in parts:
            cand = buf + sep + p if buf else p
            if len(cand) <= size: buf = cand
            else:
                if buf: out.append(buf.strip())
                buf = _split(p, seps[1:])[0] if len(p) > size else p
        if buf.strip(): out.append(buf.strip())
        return out
    raw = _split(text, seps)
    return [(r + " " + raw[i+1][:overlap]).strip()
            if i+1 < len(raw) and overlap else r
            for i, r in enumerate(raw)]


# ── Run it ─────────────────────────────────────────────────────
if __name__ == "__main__":

    DOCS = {
        "changelog_v2.4.md": """
v2.4.0 released May 2026. New features include smart filters for tasks by assignee,
due date, label, or custom field. Bulk edit lets you update status or priority across
multiple tasks at once. Full keyboard navigation added, press question mark for shortcuts.
Bug fixes: notifications not clearing on mobile, date picker wrong month on negative timezones,
duplicate webhook events on task creation.
Performance: dashboard load 40 percent faster, sync latency down from 800ms to 120ms.
        """,
        "api_reference.md": """
Authentication requires a Bearer token in the Authorization header.
Tokens are created in Settings under API Keys, scoped to read-only or read-write.
Create a task: POST /api/v1/tasks. Required fields are title and project_id.
Optional: assignee_id, due_date in ISO 8601 format, priority as low medium high or urgent.
List tasks: GET /api/v1/tasks. Filter by project_id or status.
Status values: open, in_progress, done, cancelled. Default limit 25, max 100.
        """,
        "feature_guide.md": """
Projects are top-level containers for work with their own task lists and members.
Create a project from the left sidebar New Project button. Set name, description, color, icon.
Visibility is either private for invite-only or workspace for all members.
Invite members in project Settings under the Members tab. Enter email and choose a role.
Viewer can see tasks only. Member can create and edit. Admin has full access.
Invitations expire after 7 days. Workflows define status columns: Backlog, In Progress, Done.
Customise workflows in Settings under Workflow.
        """,
    }

    # ── First run: index everything ───────────────────────────
    client     = get_client("./chroma_db")
    collection = get_collection(client, "support_docs")

    existing = collection.count()
    if existing == 0:
        print("Fresh collection — indexing all documents...")
        for filename, content in DOCS.items():
            index_document(collection, content.strip(), source=filename)
    else:
        print(f"Collection already has {existing} chunks — skipping re-index.")
        print("(Delete ./chroma_db/ folder to re-index from scratch)")

    total = collection.count()
    print(f"\nCollection '{collection.name}': {total} chunks total")

    # ── Search queries ────────────────────────────────────────
    queries = [
        "How do I invite someone to my project?",
        "What changed in the latest release?",
        "How do I create a task via the API?",
        "What is the admin role?",
        "Why is my dashboard slow?",
    ]

    print("\nRunning queries against ChromaDB...")
    for q in queries:
        results = search(collection, q, top_k=2)
        print_results(q, results)

    # ── Show collection info ──────────────────────────────────
    print(f"\n\nCollection stats:")
    print(f"  Name   : {collection.name}")
    print(f"  Count  : {collection.count()} chunks")
    print(f"  Space  : cosine (HNSW)")
    print(f"  Stored : ./chroma_db/  (persistent)")