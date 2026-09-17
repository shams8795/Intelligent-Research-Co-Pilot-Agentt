"""Q&A backend adapted from the standalone research copilot file.

This module is intentionally focused on the question-answer card workflow.
"""

from __future__ import annotations

import base64
import hashlib
import os
from io import BytesIO
from typing import Iterable

_DEPENDENCY_IMPORT_ERROR = None

try:
    import fitz
    import faiss
    import numpy as np
    from groq import Groq
    from PIL import Image
    from sentence_transformers import SentenceTransformer
except Exception as import_error:  # pragma: no cover - environment dependent
    _DEPENDENCY_IMPORT_ERROR = str(import_error)


_TEXT_MODEL = None
_CLIP_MODEL = None
_GROQ_CLIENT = None

_CACHE = {
    "signature": None,
    "chunks": [],
    "images": [],
    "index": None,
}


def _ensure_models() -> tuple[bool, str]:
    global _TEXT_MODEL, _CLIP_MODEL, _GROQ_CLIENT

    if _DEPENDENCY_IMPORT_ERROR:
        return False, (
            "Q&A module dependencies are missing. Install: pymupdf faiss-cpu numpy "
            "sentence-transformers groq pillow"
        )

    if not os.getenv("GROQ_API_KEY"):
        return False, "GROQ_API_KEY is missing. Add it to your .env file and restart."

    if _TEXT_MODEL is None:
        _TEXT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

    if _CLIP_MODEL is None:
        _CLIP_MODEL = SentenceTransformer("clip-ViT-B-32")

    if _GROQ_CLIENT is None:
        _GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))

    return True, ""


def _chunk_text(text: str, max_words: int = 250) -> list[str]:
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _extract_pdf_content(pdf_bytes: bytes) -> tuple[str, list]:
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    merged_text = ""
    images = []

    for page in document:
        merged_text += page.get_text()
        for img in page.get_images(full=True):
            xref = img[0]
            pix = fitz.Pixmap(document, xref)
            if pix.n < 5:
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            else:
                pix = fitz.Pixmap(fitz.csRGB, pix)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(image)

    document.close()
    return merged_text, images


def _build_signature(uploaded_files: Iterable) -> str:
    hasher = hashlib.sha256()
    for uploaded_file in uploaded_files:
        if hasattr(uploaded_file, "getvalue"):
            file_bytes = uploaded_file.getvalue()
        else:
            with open(uploaded_file, "rb") as file_handle:
                file_bytes = file_handle.read()

        hasher.update(file_bytes)
        hasher.update(b"||")

    return hasher.hexdigest()


def _process_uploaded_files(uploaded_files: list) -> tuple[bool, str]:
    try:
        signature = _build_signature(uploaded_files)
        if _CACHE["signature"] == signature and _CACHE["index"] is not None:
            return True, ""

        all_chunks = []
        all_images = []

        for uploaded_file in uploaded_files:
            if hasattr(uploaded_file, "getvalue"):
                file_bytes = uploaded_file.getvalue()
            else:
                with open(uploaded_file, "rb") as file_handle:
                    file_bytes = file_handle.read()

            text, images = _extract_pdf_content(file_bytes)
            all_chunks.extend(_chunk_text(text))
            all_images.extend(images)

        if not all_chunks:
            return False, "No text was extracted from the uploaded PDFs."

        embeddings = _TEXT_MODEL.encode(all_chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))

        _CACHE["signature"] = signature
        _CACHE["chunks"] = all_chunks
        _CACHE["images"] = all_images
        _CACHE["index"] = index

        return True, ""
    except Exception as error:
        return False, f"Failed to process uploaded PDFs: {error}"


def _retrieve_text(question: str, k: int = 3) -> tuple[list[str], list[float]]:
    question_vector = _TEXT_MODEL.encode([question])
    distances, indices = _CACHE["index"].search(question_vector, k)
    chunks = [_CACHE["chunks"][i] for i in indices[0] if i < len(_CACHE["chunks"])]
    return chunks, distances[0]


def _rank_images(question: str, top_k: int = 2) -> list:
    if not _CACHE["images"]:
        return []

    question_vector = _CLIP_MODEL.encode([question])[0]
    similarities = []

    for idx, image in enumerate(_CACHE["images"]):
        image_vector = _CLIP_MODEL.encode(image)
        similarity = float(np.dot(question_vector, image_vector))
        similarities.append((idx, similarity))

    similarities.sort(key=lambda item: item[1], reverse=True)
    return [_CACHE["images"][idx] for idx, _ in similarities[:top_k]]


def _analyze_image(image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    try:
        response = _GROQ_CLIENT.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this scientific plot or figure and explain the key insights.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
        )
        return str(response.choices[0].message.content)
    except Exception as error:
        return f"Image analysis failed: {error}"


def _reason_over_context(question: str, text_context: str, image_context: str) -> str:
    response = _GROQ_CLIENT.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a scientific research reasoning agent. "
                    "Use the provided evidence to answer accurately and clearly."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Text Evidence:\n{text_context}\n\n"
                    f"Image Evidence:\n{image_context}\n\n"
                    "Produce a detailed scientific answer."
                ),
            },
        ],
    )
    return str(response.choices[0].message.content)


def answer_question(question: str, uploaded_files: list | None) -> str:
    """Answer question from uploaded paper files for the Q&A card."""
    cleaned_question = (question or "").strip()
    if not cleaned_question:
        return "Please provide a valid question."

    if not uploaded_files:
        return "Upload papers first so answers can be grounded in document content."

    ready, status_message = _ensure_models()
    if not ready:
        return status_message

    processed, process_message = _process_uploaded_files(uploaded_files)
    if not processed:
        return process_message

    try:
        text_chunks, distances = _retrieve_text(cleaned_question)
        text_context = "\n\n".join(text_chunks)

        top_images = _rank_images(cleaned_question)
        image_insights = [_analyze_image(image) for image in top_images]
        image_context = "\n".join(image_insights) if image_insights else "No relevant figures found."

        answer = _reason_over_context(cleaned_question, text_context, image_context)
        avg_distance = float(np.mean(distances)) if len(distances) > 0 else 1.0
        confidence = round(max(0.1, 1 - avg_distance), 3)

        return f"{answer}\n\nConfidence: {confidence}"
    except Exception as error:
        return f"Q&A pipeline failed: {error}"
