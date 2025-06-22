from json import load
from mcp.server.fastmcp import FastMCP
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from dotenv import load_dotenv

load_dotenv(override=True)

mcp = FastMCP("vectorize_mcp")

EMBEDDING_MODEL = "nomic-embed-text:latest"
CHROMA_DB_DIR = os.environ.get("CHROMA_DB_PATH", "./")

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

@mcp.tool()
def query_user_collection(query: str, collection_name: str, top_k: int = 5) -> list[str]:
    """
    Query a specific user's PDF collection for relevant text.

    Args:
        query (str): The search query.
        collection_name (str): The name of the collection (e.g., user123_report1).
        top_k (int): Number of top results.

    Returns:
        List[str]: Top matching chunks of text.
    """
    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=CHROMA_DB_DIR,
    )

    docs = vector_store.similarity_search(query, k=top_k)
    return [doc.page_content for doc in docs]


@mcp.tool()
def vectorize_pdf(pdf_path: str, collection_name: str) -> str:
    """
    Vectorize the given PDF and store in a dedicated Chroma collection.

    Args:
        pdf_path (str): Absolute path to the PDF file.
        collection_name (str): A unique name to identify the collection (e.g., filename, user_id_timestamp, etc.)

    Returns:
        str: Absolute path to the vectorstore directory.
    """
    return _process_and_vectorize(pdf_path, collection_name)


def _process_and_vectorize(pdf_path: str, collection_name: str) -> str:
    if not os.path.isfile(pdf_path) or not pdf_path.lower().endswith(".pdf"):
        raise ValueError("Provided path must be a valid .pdf file")

    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=CHROMA_DB_DIR,
    )

    vector_store.add_documents(chunks)
    vector_store.persist()

    return os.path.abspath(os.path.join(CHROMA_DB_DIR, f"chroma-{collection_name}"))

if __name__ == "__main__":
    mcp.run()
