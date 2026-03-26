"""ChromaDB vector store wrapper for document embeddings."""

import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from rag.chunker import chunk_documents
from config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    TOP_K,
)


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Initialize Gemini embedding model."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


def get_vector_store() -> Chroma:
    """
    Get or create a persistent ChromaDB vector store.

    Returns a LangChain Chroma instance backed by the Gemini embedding
    model, persisted to disk at the path specified in config.
    """
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=_get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def add_documents(documents: list[Document]) -> Chroma:
    """
    Chunk and embed documents, then store in ChromaDB.

    Args:
        documents: List of LangChain Document objects to process.

    Returns:
        The updated Chroma vector store instance.
    """
    chunks = chunk_documents(documents)
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    return vector_store


def get_retriever(k: int = TOP_K):
    """
    Get a LangChain retriever from the vector store.

    Args:
        k: Number of top results to retrieve.

    Returns:
        A retriever that searches ChromaDB for relevant chunks.
    """
    vector_store = get_vector_store()
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def delete_collection():
    """Delete all documents from the vector store."""
    vector_store = get_vector_store()
    vector_store.delete_collection()
