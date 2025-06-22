from mcp.server.fastmcp import FastMCP
import chromadb
from chromadb.config import Settings
import httpx
import os
from typing import Dict, List, Optional

# Initialize FastMCP app
app = FastMCP("vector_store_mcp")

# Initialize ChromaDB
CHROMA_PATH = "data/vectorstore.db"
os.makedirs(os.path.dirname(CHROMA_PATH), exist_ok=True)

try:
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(
            allow_reset=True,
            is_persistent=True
        )
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize ChromaDB client: {e}")

# Create or retrieve collection
try:
    collection = client.get_or_create_collection(
        name="text_vectors",
        metadata={"hnsw:space": "cosine"}
    )
except Exception as e:
    raise RuntimeError(f"Failed to create or retrieve ChromaDB collection: {e}")

def get_embedding(text: str) -> List[float]:
    """Get embeddings from Ollama API."""
    try:
        print(f"Requesting embedding for text: {text[:50]}...")
        response = httpx.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": "nomic-embed-text:v1.5",
                "prompt": text
            },
            timeout=30.0  # Increase timeout
        )
        response.raise_for_status()

        # Log the full response
        response_data = response.json()
        print(f"Response status: {response.status_code}")
        
        # Handle different possible response formats
        if "embedding" in response_data:
            return response_data["embedding"]
        elif "embeddings" in response_data:
            return response_data["embeddings"]
        elif "data" in response_data and len(response_data["data"]) > 0:
            if "embedding" in response_data["data"][0]:
                return response_data["data"][0]["embedding"]
            
        # If response contains a list directly
        if isinstance(response_data, list) and len(response_data) > 0:
            if isinstance(response_data[0], float):
                return response_data
        
        print(f"Unexpected response format: {response_data}")
        raise RuntimeError(f"Could not find embeddings in response: {response_data}")
        
    except httpx.RequestError as e:
        print(f"Request error: {e}")
        raise RuntimeError(f"Failed to fetch embeddings from Ollama API: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise RuntimeError(f"Error processing embeddings: {e}")

@app.tool()
def store_vector(
    text: str,
    metadata: Optional[Dict] = None
) -> Dict:
    """
    Vectorize text and store in ChromaDB.

    Args:
        text: Text to vectorize and store.
        metadata: Optional metadata to store with the vector.

    Returns:
        Dict with database URL and schema info.
    """
    try:
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
    except Exception as e:
        raise RuntimeError(f"Failed to store vector in ChromaDB: {e}")

if __name__ == "__main__":
    print(get_embedding("test"))
