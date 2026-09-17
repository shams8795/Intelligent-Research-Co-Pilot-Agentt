"""Load PDF documents for the RAG pipeline."""

import os
import tempfile
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(uploaded_file) -> List[Document]:
    """Load an uploaded PDF and return LangChain documents.

    PyPDFLoader expects a file path, so the uploaded in-memory file is written
    temporarily to disk, loaded, then removed.
    """
    if uploaded_file is None:
        raise ValueError("No PDF file was uploaded.")

    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        raise ValueError("Uploaded PDF is empty.")

    temp_pdf_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(file_bytes)
            temp_pdf_path = temp_pdf.name

        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()

        if not documents:
            raise ValueError("No text could be extracted from this PDF.")

        # Keep a friendly source name for downstream UI and debugging.
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name

        return documents
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
