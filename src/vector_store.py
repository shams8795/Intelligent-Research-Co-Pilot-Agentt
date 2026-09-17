
"""Create and return a Chroma vector store for PDF chunks."""

from typing import List

from chromadb.utils import embedding_functions
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma


class ChromaDefaultEmbeddings(Embeddings):
    """Adapter for Chroma's built-in embedding model.

    This keeps embedding generation local to the vector DB stack and avoids
    OpenAI dependency for embeddings while still creating real vectors.
    """

    def __init__(self) -> None:
        self._embedding_function = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self._embedding_function(texts)
        # Chroma validation expects native Python numeric types (float/int),
        # not numpy scalar types like np.float32.
        return [[float(value) for value in vector] for vector in vectors]

    def embed_query(self, text: str) -> List[float]:
        return [float(value) for value in self._embedding_function([text])[0]]


def create_vector_store(
    documents: List[Document],
    collection_name: str = "research_papers",
) -> Chroma:
    """Create a Chroma vector store from split document chunks."""
    if not documents:
        raise ValueError("Documents list cannot be empty")

    embeddings = ChromaDefaultEmbeddings()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=None,
    )

    return vector_store
#http://localhost:8501