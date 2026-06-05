# ann_search.py
# HNSW (via ChromaDB) vs FAISS (IVF) — side by side benchmark
# Shows exactly what ChromaDB is doing under the hood

import time
import numpy as np
import requests
import faiss

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL      = "nomic-embed-text"
DIMS       = 768


# ── Embed helper ──────────────────────────────────────────────
def embed(text: str) -> np.ndarray:
    r = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": f"search_document: {text}"})
    r.raise_for_status()
    return np.array(r.json()["embedding"], dtype=np.float32)


# ── Generate fake vectors for benchmarking ────────────────────
def make_fake_vectors(n: int, dims: int = DIMS) -> np.ndarray:
    """Random unit-normalised vectors — simulates a real embedding index."""
    vecs = np.random.randn(n, dims).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms   # unit normalise for cosine


# ── 1. Brute force (what index.json did) ─────────────────────
def brute_force_search(index_vecs: np.ndarray, query_vec: np.ndarray, top_k: int = 5):
    dots = index_vecs @ query_vec          # dot product = cosine sim when normalised
    top_k_idx = np.argpartition(dots, -top_k)[-top_k:]
    return top_k_idx[np.argsort(dots[top_k_idx])[::-1]]


# ── 2. FAISS — IVFFlat (cluster-based ANN) ───────────────────
def build_faiss_index(vecs: np.ndarray, n_lists: int = 32) -> faiss.Index:
    """
    IVFFlat: clusters vectors into n_lists Voronoi cells.
    At query time, only searches n_probe closest cells.

    n_lists rule of thumb: sqrt(N) for datasets up to 1M
    """
    quantizer = faiss.IndexFlatIP(DIMS)   # inner product (cosine when normalised)
    index = faiss.IndexIVFFlat(quantizer, DIMS, n_lists, faiss.METRIC_INNER_PRODUCT)
    index.train(vecs)      # clustering happens here
    index.add(vecs)
    return index

def faiss_search(index: faiss.Index, query_vec: np.ndarray, top_k: int = 5, n_probe: int = 4):
    """
    n_probe: how many clusters to search.
    Higher = more accurate but slower.
    Production default: n_probe = sqrt(n_lists)
    """
    index.nprobe = n_probe
    q = query_vec.reshape(1, -1)
    distances, indices = index.search(q, top_k)
    return indices[0], distances[0]


# ── 3. FAISS — HNSW (same algorithm as ChromaDB) ─────────────
def build_faiss_hnsw(vecs: np.ndarray, M: int = 16) -> faiss.Index:
    """
    HNSW in FAISS.
    M: number of connections per node per layer.
    Higher M = better recall, more memory.
    ChromaDB default M=16, ef_construction=100.
    """
    index = faiss.IndexHNSWFlat(DIMS, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 100   # higher = better graph, slower build
    index.add(vecs)
    return index

def hnsw_search(index: faiss.Index, query_vec: np.ndarray, top_k: int = 5, ef: int = 50):
    """
    ef: beam width at search time.
    Higher ef = more accurate, slightly slower.
    ChromaDB default: ef=10
    """
    index.hnsw.efSearch = ef
    q = query_vec.reshape(1, -1)
    distances, indices = index.search(q, top_k)
    return indices[0], distances[0]


# ── Benchmark ─────────────────────────────────────────────────
def benchmark(n_vectors: int = 100_000, n_queries: int = 50, top_k: int = 5):
    print(f"\nBenchmark: {n_vectors:,} vectors × {n_queries} queries × top-{top_k}")
    print("=" * 58)

    vecs  = make_fake_vectors(n_vectors)
    query = make_fake_vectors(1)[0]

    # Build indexes
    print("\nBuilding indexes...")
    t0 = time.time(); faiss_ivf  = build_faiss_index(vecs, n_lists=int(np.sqrt(n_vectors))); print(f"  FAISS IVF    built in {time.time()-t0:.2f}s")
    t0 = time.time(); faiss_hnsw = build_faiss_hnsw(vecs, M=16);                              print(f"  FAISS HNSW   built in {time.time()-t0:.2f}s")
    print(f"  Brute force  no build needed")

    # Run queries
    print("\nSearch speed (avg over queries):")

    t0 = time.time()
    for _ in range(n_queries):
        brute_force_search(vecs, query, top_k)
    bf_ms = (time.time() - t0) / n_queries * 1000
    print(f"  Brute force  : {bf_ms:.2f} ms/query   (checks ALL {n_vectors:,} vectors)")

    t0 = time.time()
    for _ in range(n_queries):
        faiss_search(faiss_ivf, query, top_k, n_probe=int(np.sqrt(np.sqrt(n_vectors))))
    ivf_ms = (time.time() - t0) / n_queries * 1000
    print(f"  FAISS IVF    : {ivf_ms:.2f} ms/query   (checks ~{int(np.sqrt(n_vectors))} vectors)")

    t0 = time.time()
    for _ in range(n_queries):
        hnsw_search(faiss_hnsw, query, top_k, ef=50)
    hnsw_ms = (time.time() - t0) / n_queries * 1000
    print(f"  FAISS HNSW   : {hnsw_ms:.2f} ms/query   (navigates graph layers)")

    print(f"\nSpeedup over brute force:")
    print(f"  IVF  : {bf_ms/ivf_ms:.1f}x faster")
    print(f"  HNSW : {bf_ms/hnsw_ms:.1f}x faster")

    # Recall check — how many top-5 results match brute force?
    true_ids = set(brute_force_search(vecs, query, top_k))
    ivf_ids  = set(faiss_search(faiss_ivf,  query, top_k)[0])
    hnsw_ids = set(hnsw_search(faiss_hnsw, query, top_k)[0])
    print(f"\nRecall@{top_k} (vs brute force ground truth):")
    print(f"  IVF  : {len(true_ids & ivf_ids)}/{top_k}  ({len(true_ids & ivf_ids)/top_k*100:.0f}%)")
    print(f"  HNSW : {len(true_ids & hnsw_ids)}/{top_k}  ({len(true_ids & hnsw_ids)/top_k*100:.0f}%)")


if __name__ == "__main__":
    benchmark(n_vectors=100_000,  n_queries=50)
    benchmark(n_vectors=100_000, n_queries=20)