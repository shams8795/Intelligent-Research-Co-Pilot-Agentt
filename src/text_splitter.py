"""Split loaded PDF documents into chunks for embedding and retrieval."""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int = 1200,
    chunk_overlap: int = 75,
) -> List[Document]:
    """Split documents into context-preserving chunks."""
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    return text_splitter.split_documents(documents)
