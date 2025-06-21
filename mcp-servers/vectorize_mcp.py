from mcp.server.fastmcp import FastMCP
import chromadb
from chromadb.config import Settings
import httpx
import os
from typing import Dict, List, Optional

app = FastMCP("vector_store_mcp")

# Initialize ChromaDB
CHROMA_PATH = "data/vectorstore.db"
os.makedirs(os.path.dirname(CHROMA_PATH), exist_ok=True)

client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(
        allow_reset=True,
        is_persistent=True
    )
)

# Create collection
collection = client.get_or_create_collection(
    name="text_vectors",
    metadata={"hnsw:space": "cosine"}
)

def get_embedding(text: str) -> List[float]:
    """Get embeddings from Ollama API"""
    response = httpx.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text:v1.5",
            "prompt": text
        }
    )
    return response.json()["embeddings"]

@app.tool()
def store_vector(
    text: str,
    metadata: Optional[Dict] = None
) -> Dict:
    """
    Vectorize text and store in ChromaDB.

    Args:
        text: Text to vectorize and store
        metadata: Optional metadata to store with the vector

    Returns:
        Dict with database URL and schema info
    """
    # Generate embedding
    embedding = get_embedding(text)
    
    # Store in ChromaDB
    collection.add(
        embeddings=[embedding],
        documents=[text],
        metadatas=[metadata or {}],
        ids=[str(hash(text))]
    )

    return {
        "database_url": f"file://{os.path.abspath(CHROMA_PATH)}",
        "schema": {
            "collection": "text_vectors",
            "vector_dimension": len(embedding),
            "metadata_schema": "dynamic",
            "distance_metric": "cosine"
        }
    }

if __name__ == "__main__":
    app.run()
