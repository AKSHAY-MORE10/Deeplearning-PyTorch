import requests
import numpy as np
from chunker import chunk_recursive, Chunk

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"

def embed(text: str) -> np.ndarray:
    r = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": f"search_document: {text}"})
    r.raise_for_status()
    return np.array(r.json()["embedding"])

# A realistic document
document = """
Python is a high-level programming language known for its simplicity.
It was created by Guido van Rossum and released in 1991.

Python supports multiple programming paradigms including object-oriented,
functional, and procedural styles. It has a large standard library.

Machine learning frameworks like PyTorch and TensorFlow are written in Python.
Most AI research code is Python-first due to its ecosystem.

FastAPI is a modern Python web framework for building APIs quickly.
It is built on top of Starlette and Pydantic for validation.
"""

# Step 1: Chunk
chunks = chunk_recursive(document, chunk_size=200, overlap=40)
print(f"Document split into {len(chunks)} chunks\n")

# Step 2: Embed each chunk
print("Embedding chunks...")
chunk_embeddings = []
for c in chunks:
    vec = embed(c.text)
    chunk_embeddings.append(vec)
    print(f"  Chunk {c.index+1}: '{c.text[:50]}...' → {len(vec)} dims")

# Step 3: Query it (manual cosine search — Topic 3 will formalise this)
from sklearn.metrics.pairwise import cosine_similarity

query = "search_query: What is Python used for in AI?"
query_vec = np.array(requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": query}).json()["embedding"])

scores = cosine_similarity([query_vec], chunk_embeddings)[0]
ranked = sorted(zip(scores, chunks), reverse=True)

print(f"\nQuery: '{query}'\n")
print("Top matching chunks:")
for score, chunk in ranked[:3]:
    print(f"  {score:.3f}  →  {chunk.text[:80]}...")