"""RAG summarization orchestrator using Groq via LangChain ChatGroq."""

import os
from typing import List
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from src.ollama_handler import is_ollama_running, summarize_with_ollama
from src.pdf_loader import load_pdf
from src.retriever import create_retriever
from src.text_splitter import split_documents
from src.vector_store import create_vector_store


load_dotenv()

SUMMARY_RETRIEVAL_QUERIES = [
    "problem statement research objective and core contribution",
    "proposed method technical approach and key innovation",
    "experimental results key metrics and performance compared to baselines",
    "real-world applications and impact limitations and future work",
]

SUMMARY_RETRIEVAL_K = 3
SUMMARY_RETRIEVAL_FETCH_K = 12
MAX_RETRIEVED_CONTEXT_CHARS = 35_000


def _ensure_groq_api_key() -> None:
    """Validate that the Groq key exists in environment or .env."""
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your .env file and restart Streamlit."
        )


def _get_groq_model_name() -> str:
    """Return configured Groq model name, or a supported default."""
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def _build_summary_prompt(context: str) -> str:
    """Build a detailed technical prompt using retrieved paper context only."""
    return (
        "You are an expert AI researcher and paper reviewer.\n"
        "Your task is NOT just to summarize the paper, but to produce a deep, insight-driven, technically strong analysis.\n"
        "Do NOT write generic or descriptive summaries. Explain WHY and HOW, not just WHAT.\n\n"
        "Rules:\n"
        "- Use precise, technical, and assertive language.\n"
        "- Do not restate what the paper says — explain WHY and HOW.\n"
        "- Avoid vague phrases like 'improves performance' or 'handles dependencies'.\n"
        "- Use formal academic tone.\n"
        "- Be direct and do not repeat ideas across sections.\n"
        "- Do not copy phrasing from the paper. Rewrite in plain, precise language.\n"
        "- Do not invent missing details. If something is not supported by the context, leave it out.\n"
        "- Include only the metrics that actually appear in the paper. Do not assume AUC, F1, or any other metric unless explicitly reported.\n"
        "- If there are multiple evaluation measures, focus on the most important ones used by the paper.\n"
        "- Do not miss key reported metrics. If the paper includes critical results, include them when they are actually present.\n"
        "- If the paper reports no numerical results, describe the performance qualitatively without inventing numbers.\n"
        "- Keep the response concise, technically grounded, and information-dense.\n"
        "- Use short bullet points where they clarify technical concepts.\n\n"
        "Output format:\n"
        "Use ONLY these section titles, in this exact order:\n"
        "1. Problem Statement\n"
        "2. Proposed Solution\n"
        "3. Key Insight ⭐ (MOST IMPORTANT)\n"
        "4. Important Results\n"
        "5. Real-World Impact\n"
        "6. Limitations\n\n"
        "Section Guidance:\n"
        "- Problem Statement: Explain what problem the paper addresses and WHY baseline methods are insufficient for this specific challenge.\n"
        "- Proposed Solution: Focus on what is NEW or different in the technical approach. Do not describe standard components.\n"
        "- Key Insight: Explain the core technical advantage behind the method and WHY it works better than previous approaches.\n"
        "- Important Results: Extract the paper's actual evaluation metrics and include specific values when available.\n"
        "- Real-World Impact: Describe concrete, technical impact.\n"
        "- Limitations: Identify real technical limitations.\n\n"
        "Each section must reflect deep understanding and broader technical context.\n"
        "---\n"
        "RETRIEVED PAPER CONTEXT:\n"
        f"{context}"
    )


def _build_collection_name(uploaded_file) -> str:
    """Create a unique in-memory collection name for each summarization run."""
    source_name = getattr(uploaded_file, "name", "paper") or "paper"
    safe_source = "".join(ch if ch.isalnum() else "_" for ch in source_name.lower())
    safe_source = safe_source.strip("_") or "paper"
    safe_source = safe_source[:20]
    return f"{safe_source}_{uuid4().hex[:8]}"


def _sort_documents_by_page(documents: List[Document]) -> List[Document]:
    """Sort retrieved chunks by page so the final context reads in paper order."""

    def _page_number(document: Document) -> int:
        page = document.metadata.get("page", 10**9)
        try:
            return int(page)
        except (TypeError, ValueError):
            return 10**9

    return sorted(documents, key=_page_number)


def _retrieve_summary_documents(vector_store) -> List[Document]:
    """Retrieve diverse, high-signal chunks needed for a technical paper summary."""
    retriever = create_retriever(
        vector_store,
        k=SUMMARY_RETRIEVAL_K,
        fetch_k=SUMMARY_RETRIEVAL_FETCH_K,
        search_type="mmr",
    )

    retrieved_documents: List[Document] = []
    seen_chunks = set()

    for query in SUMMARY_RETRIEVAL_QUERIES:
        for document in retriever.invoke(query):
            content = document.page_content.strip()
            if not content:
                continue

            chunk_key = (
                str(document.metadata.get("source", "")),
                str(document.metadata.get("page", "")),
                content,
            )
            if chunk_key in seen_chunks:
                continue

            seen_chunks.add(chunk_key)
            retrieved_documents.append(document)

    if not retrieved_documents:
        raise ValueError("No relevant chunks were retrieved from the PDF.")

    return _sort_documents_by_page(retrieved_documents)


def _build_retrieved_context(documents: List[Document]) -> str:
    """Convert retrieved chunks into the compact context sent to the LLM."""
    context_parts: List[str] = []
    current_size = 0

    for index, document in enumerate(documents, start=1):
        content = document.page_content.strip()
        if not content:
            continue

        page = document.metadata.get("page", "unknown")
        chunk_text = f"[Retrieved chunk {index} | page {page}]\n{content}"
        projected_size = current_size + len(chunk_text) + 2

        if context_parts and projected_size > MAX_RETRIEVED_CONTEXT_CHARS:
            break

        context_parts.append(chunk_text)
        current_size = projected_size

    context = "\n\n".join(context_parts).strip()
    if not context:
        raise ValueError("Retrieved context was empty after chunk filtering.")

    return context


def build_retrieved_context_for_pdf(uploaded_file) -> str:
    """Reusable helper that builds retrieved context for one PDF."""
    if uploaded_file is None:
        raise ValueError("Please upload a PDF before processing.")

    documents = load_pdf(uploaded_file)

    chunks = split_documents(documents, chunk_size=1500, chunk_overlap=100)
    if not chunks:
        raise ValueError("No readable text chunks were produced from this PDF.")

    vector_store = create_vector_store(
        chunks,
        collection_name=_build_collection_name(uploaded_file),
    )

    retrieved_documents = _retrieve_summary_documents(vector_store)
    return _build_retrieved_context(retrieved_documents)


def _generate_summary_with_model(summary_context: str) -> str:
    """Try Groq first, then fallback to Ollama if Groq rate limit is hit."""
    try:
        llm = ChatGroq(
            model=_get_groq_model_name(),
            temperature=0,
        )
        response = llm.invoke(_build_summary_prompt(summary_context))

        if hasattr(response, "content") and response.content is not None:
            summary_text = str(response.content).strip()
        else:
            summary_text = str(response).strip()

        if not summary_text:
            raise ValueError("The model returned an empty summary. Please retry.")

        return summary_text

    except Exception as groq_error_obj:
        groq_error = str(groq_error_obj)
        if "Rate limit" in groq_error or "429" in groq_error or "tokens per day" in groq_error.lower():
            if is_ollama_running():
                summary_text = summarize_with_ollama(
                    summary_context,
                    _build_summary_prompt(""),
                    model="mistral",
                )
                if summary_text:
                    return f"📌 GENERATED WITH LOCAL AI (Groq API rate limited)\n\n{summary_text}"

            raise ValueError(
                "🚨 API Rate Limit Reached (Groq)\n\n"
                "The daily token limit for Groq API has been exceeded.\n\n"
                "To continue, you have two options:\n"
                "1. Wait for the limit to reset in ~24 hours\n"
                "2. Install Ollama for offline AI: https://ollama.com/download\n"
                "   Then run: ollama pull mistral"
            )

        raise groq_error_obj


def summarize_pdf(uploaded_file) -> str:
    """Generate a PDF summary through a full RAG pipeline."""
    if uploaded_file is None:
        raise ValueError("Please upload a PDF before generating a summary.")

    _ensure_groq_api_key()
    summary_context = build_retrieved_context_for_pdf(uploaded_file)
    return _generate_summary_with_model(summary_context)


def summarize_multiple_pdfs(uploaded_files: List) -> dict:
    """Generate summaries for multiple PDF files (max 5)."""
    if not uploaded_files:
        raise ValueError("Please upload at least one PDF before generating summaries.")

    if len(uploaded_files) > 5:
        raise ValueError("Maximum 5 PDF files allowed. Please upload fewer files.")

    _ensure_groq_api_key()

    summaries = {}
    for uploaded_file in uploaded_files:
        try:
            summary_context = build_retrieved_context_for_pdf(uploaded_file)
            summaries[uploaded_file.name] = _generate_summary_with_model(summary_context)
        except Exception as error:
            summaries[uploaded_file.name] = f"Error generating summary: {error}"

    return summaries