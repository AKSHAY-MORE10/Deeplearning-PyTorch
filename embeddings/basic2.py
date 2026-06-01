import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"


def embed(text: str):
    response = requests.post(OLLAMA_URL, json={
        "model" :MODEL,
        "prompt":text
    })

    response.raise_for_status()
    vector = response.json()["embedding"]
    return np.array(vector)

def embed_batch(texts: list[str]):
    return np.array([embed(t) for t in texts])

def similarity(vec1: np.ndarray, vec2: np.ndarray):
    return cosine_similarity([vec1], [vec2])[0][0]

if __name__ == "__main__":
    texts = [
        "Python is a programming language",
        "Java is used for backend development",
        "I love eating pizza",
        "Pasta is my favourite Italian food",
        "The sun rises in the east",
        "Stars are visible at night",
    ]

    print("generated embeding")
    embeddings = embed_batch(texts)

    print(f"Model output shape : {embeddings.shape}")   # (6, 768)
    print(f"Each vector has    : {embeddings.shape[1]} dimensions\n")

    sim = cosine_similarity(embeddings)
    print("Similarity matrix (higher = more similar):\n")
    print(f"{'':>42}", end="")
    for t in texts:
        print(f"  {t[:12]:>12}", end="")
    print()

    for i, t1 in enumerate(texts):
        print(f"{t1[:40]:>42}", end="")
        for j in range(len(texts)):
            print(f"  {sim[i][j]:>12.3f}", end="")
        print()

    # Cleaner view — just the interesting pairs
    print("\n── Pair-wise similarities ──")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = sim[i][j]
            bar = "█" * int(score * 20)
            print(f"  {score:.3f} {bar:<20}  '{texts[i][:28]}' ↔ '{texts[j][:28]}'")


def embed_query(text: str) -> np.ndarray:
    """For search queries — use search_query prefix."""
    return embed(f"search_query: {text}")

def embed_document(text: str) -> np.ndarray:
    """For documents/passages — use search_document prefix."""
    return embed(f"search_document: {text}")

# Example — this is how RAG will work later
query_vec = embed_query("What programming languages exist?")
doc_vec   = embed_document("Python is a popular high-level language.")

print(f"Query-doc similarity: {similarity(query_vec, doc_vec):.3f}")