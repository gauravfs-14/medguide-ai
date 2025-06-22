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


@mcp.tool()
def vectorize_pdf(
    pdf_path: str,
) -> str:
    """
    Load a PDF, extract text, vectorize it, and return Chroma DB path.

    Args:
        pdf_path (str): Absolute path to the PDF file.

    Returns:
        str: The local path where Chroma DB is stored.
    """
    return _process_and_vectorize(pdf_path)


def _process_and_vectorize(pdf_path: str) -> str:
    if not os.path.isfile(pdf_path) or not pdf_path.lower().endswith(".pdf"):
        raise ValueError("Provided path must be a valid .pdf file")

    # Step 1: Load PDF
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    # Step 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Step 3: Embed and store
    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name="pdf_collection",
        embedding_function=embedding_function,
        persist_directory=CHROMA_DB_DIR,
    )

    vector_store.add_documents(chunks)
    vector_store.persist()

    return os.path.abspath(CHROMA_DB_DIR)

if __name__ == "__main__":
    mcp.run()
