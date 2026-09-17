import os
import json
import re
import tempfile
from typing import List, Dict, Any

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

# =============================
# LLM + Embedding Models
# =============================
def _get_llm():
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(model=model, temperature=0.2)


_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _embed(text: str):
    return get_embedder().encode(text)




# =============================
# PDF extraction
# =============================
def _extract_text_from_pdf(uploaded_file) -> str:
    uploaded_file.seek(0)

    # create temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text = "\n".join([d.page_content for d in docs])

    finally:
        # cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return text

# =============================
# Structured Paper Extraction (REAL)
# =============================
def _extract_structured_paper(text: str) -> Dict[str, Any]:
    llm = _get_llm()

    prompt = f"""
You are a research paper parser.

Extract structured JSON ONLY:

{{
  "title": "",
  "problem": "",
  "methods": [],
  "datasets": [],
  "limitations": [],
  "future_work": []
}}

Rules:
- Return ONLY valid JSON
- No explanation
- No markdown

PAPER:
{text[:12000]}
"""

    response = llm.invoke(prompt).content

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return {
            "title": "Unknown",
            "problem": "",
            "methods": [],
            "datasets": [],
            "limitations": [],
            "future_work": [],
        }

    return json.loads(match.group(0))


# =============================
# Signal Extraction
# =============================
def _extract_signals(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_limitations = []
    all_methods = []
    all_datasets = []

    for p in papers:
        all_limitations += p.get("limitations", [])
        all_methods += p.get("methods", [])
        all_datasets += p.get("datasets", [])

    return {
        "limitations": all_limitations,
        "methods": all_methods,
        "datasets": all_datasets,
    }


# =============================
# Semantic Gap Detection (REAL)
# =============================
def _semantic_gap_detection(papers: List[Dict]) -> List[Dict]:
    gaps = []

    all_limitations = []
    all_methods = []

    for p in papers:
        all_limitations += p.get("limitations", [])
        all_methods += p.get("methods", [])

    # embed limitations
    if len(all_limitations) > 0:
        gaps.append({
            "type": "limitation_gap",
            "description": "Cross-paper shared limitations indicate underexplored problem areas",
            "evidence": all_limitations[:3]
        })

    # method diversity check
    if len(all_methods) < len(papers):
        gaps.append({
            "type": "method_gap",
            "description": "Limited methodological diversity across papers",
            "evidence": all_methods[:3]
        })

    # dataset gap
    all_datasets = []
    for p in papers:
        all_datasets += p.get("datasets", [])

    if len(set(all_datasets)) < 2:
        gaps.append({
            "type": "dataset_gap",
            "description": "Insufficient dataset diversity across studies",
            "evidence": all_datasets[:3]
        })

    return gaps


# =============================
# Hypothesis Generation (RAG style)
# =============================
def _generate_hypothesis_from_gap(papers, gap):
    llm = _get_llm()

    context = "\n\n".join([
    f"""
TITLE: {p.get('title', '')}

PROBLEM:
{p.get('problem', '')}

METHODS:
{', '.join(p.get('methods', []))}

DATASETS:
{', '.join(p.get('datasets', []))}

LIMITATIONS:
{', '.join(p.get('limitations', []))}

FUTURE WORK:
{', '.join(p.get('future_work', []))}
"""
    for p in papers
])

    prompt = f"""
You are a top-tier ML researcher writing NeurIPS-level hypotheses.

Use ONLY the provided paper evidence.

GAP:
{gap['description']}

EVIDENCE:
{context}

Return ONLY valid JSON:

{{
  "hypothesis": "",
  "based_on_gap": "",
  "reasoning": "",
  "expected_improvement": "",
  "novelty_rationale": ""
}}
"""

    response = llm.invoke(prompt).content

    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return {
            "hypothesis": "Failed to generate",
            "based_on_gap": gap["description"],
            "reasoning": "",
            "expected_improvement": "",
            "novelty_rationale": ""
        }

    return json.loads(match.group(0))


# =============================
# Scoring (Research-grade)
# =============================
def _cosine(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b + 1e-8)


def _score_hypothesis(h, papers):
    score = 0.0

    # reasoning quality
    score += min(len(h.get("reasoning", "")) / 500, 0.25)

    # novelty (embedding-based)
    hyp_emb = _embed(h.get("hypothesis", ""))

    paper_text = " ".join([p.get("problem", "") for p in papers])
    paper_emb = _embed(paper_text)

    similarity = _cosine(hyp_emb, paper_emb)
    novelty = 1 - similarity

    score += novelty * 0.45

    # evidence grounding
    if h.get("based_on_gap"):
        score += 0.15

    # clarity
    score += min(len(h.get("expected_improvement", "")) / 300, 0.15)

    return round(score, 3)


# =============================
# MAIN PIPELINE (FINAL)
# =============================
def generate_hypotheses_v2(papers: List[Any]) -> List[Dict[str, Any]]:

    processed_papers = []

    # 1. real extraction
    for p in papers:
        text = _extract_text_from_pdf(p)
        structured = _extract_structured_paper(text)
        processed_papers.append(structured)

    # 2. semantic gaps
    gaps = _semantic_gap_detection(processed_papers)

    hypotheses = []

    # 3. generate hypotheses per gap
    for gap in gaps:
        h = _generate_hypothesis_from_gap(processed_papers, gap)
        h["score"] = _score_hypothesis(h, processed_papers)
        h["gap_type"] = gap["type"]
        hypotheses.append(h)

    return sorted(hypotheses, key=lambda x: x["score"], reverse=True)