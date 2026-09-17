import json
import os
import re
from typing import Any, Dict, List

from langchain_groq import ChatGroq

from src.schemas import PaperAnalysis


MAX_MODELS = 6
MAX_DATASETS = 6
MAX_METHODS = 6
MAX_TASKS = 4
MAX_METRICS = 6
MAX_RESULTS = 6
MAX_LIMITATIONS = 5
MAX_FUTURE_WORK = 5
MAX_CONCLUSIONS = 4
MAX_ARCH_CHANGES = 5
MAX_TRAINING_SETUP = 6
MAX_RESULT_RECORDS = 5
MAX_STRENGTHS = 5
MAX_NOVELTY = 5


def _get_llm() -> ChatGroq:
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(model=model, temperature=0)


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)

    return cleaned.strip()


def _extract_json_object(text: str) -> str:
    cleaned = _strip_code_fences(text)

    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidate = match.group(0).strip()
        json.loads(candidate)
        return candidate

    raise ValueError("No valid JSON object found in model output.")


def _clean_text(value: str) -> str:
    if not isinstance(value, str):
        return ""

    text = value.strip()

    text = text.replace("V AD", "VAD")
    text = text.replace("multi modal", "multimodal")
    text = text.replace("self supervised", "self-supervised")
    text = text.replace("semi supervised", "semi-supervised")

    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip('"').strip("'").strip()

    replacements = {
        "Crowed Violence": "Crowd Violence",
        "anomalous behavior detection": "Anomalous behavior detection",
        "video anomaly detection": "Video anomaly detection",
    }
    return replacements.get(text, text)


def _clean_list(values: Any, max_items: int) -> List[str]:
    if not isinstance(values, list):
        return []

    cleaned: List[str] = []
    seen = set()

    for item in values:
        if not isinstance(item, str):
            continue

        text = _clean_text(item)
        if not text:
            continue

        key = text.lower()
        if key in seen:
            continue

        seen.add(key)
        cleaned.append(text)

        if len(cleaned) >= max_items:
            break

    return cleaned


def _filter_generic_results(results: List[str]) -> List[str]:
    filtered = []
    generic_patterns = [
        "high anomalous scores",
        "good performance",
        "better performance",
        "effective method",
        "significant potential",
        "various methods have been proposed",
    ]

    for item in results:
        lowered = item.lower()
        if any(pattern in lowered for pattern in generic_patterns):
            continue
        filtered.append(item)

    return filtered[:MAX_RESULTS]


def _clean_entities(entities: Any) -> Dict[str, List[str]]:
    if not isinstance(entities, dict):
        entities = {}

    return {
        "models": _clean_list(entities.get("models", []), MAX_MODELS),
        "datasets": _clean_list(entities.get("datasets", []), MAX_DATASETS),
        "methods": _clean_list(entities.get("methods", []), MAX_METHODS),
        "tasks": _clean_list(entities.get("tasks", []), MAX_TASKS),
        "metrics": _clean_list(entities.get("metrics", []), MAX_METRICS),
    }


def _clean_result_records(records: Any) -> List[Dict[str, str]]:
    if not isinstance(records, list):
        return []

    cleaned: List[Dict[str, str]] = []
    seen = set()

    for item in records:
        if not isinstance(item, dict):
            continue

        record = {
            "dataset": _clean_text(item.get("dataset", "")),
            "metric": _clean_text(item.get("metric", "")),
            "value": _clean_text(item.get("value", "")),
            "setting": _clean_text(item.get("setting", "")),
            "note": _clean_text(item.get("note", "")),
        }

        if not any(record.values()):
            continue

        key = (
            record["dataset"].lower(),
            record["metric"].lower(),
            record["value"].lower(),
            record["setting"].lower(),
            record["note"].lower(),
        )
        if key in seen:
            continue

        seen.add(key)
        cleaned.append(record)

        if len(cleaned) >= MAX_RESULT_RECORDS:
            break

    return cleaned


def _infer_paper_type(title: str, proposed_method: str, entities: Dict[str, List[str]]) -> str:
    combined = f"{title} {proposed_method}".lower()

    survey_keywords = [
        "survey",
        "review",
        "taxonomy",
        "overview",
        "benchmark",
        "comparative analysis",
    ]
    if any(word in combined for word in survey_keywords):
        return "survey_paper"

    if len(entities.get("models", [])) >= 6 and len(entities.get("methods", [])) >= 4:
        return "survey_paper"

    if proposed_method.strip():
        return "method_paper"

    return "unclear"


def _normalize_analysis_data(
    data: Dict[str, Any], file_name: str, paper_id: str
) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Analysis output is not a dictionary.")

    entities = _clean_entities(data.get("entities", {}))

    normalized: Dict[str, Any] = {
        "paper_id": _clean_text(data.get("paper_id", "")) or paper_id,
        "file_name": _clean_text(data.get("file_name", "")) or file_name,
        "paper_type": _clean_text(data.get("paper_type", "")),
        "title": _clean_text(data.get("title", "")),
        "problem_statement": _clean_text(data.get("problem_statement", "")),
        "proposed_method": _clean_text(data.get("proposed_method", "")),
        "architecture_changes": _clean_list(data.get("architecture_changes", []), MAX_ARCH_CHANGES),
        "datasets": _clean_list(data.get("datasets", []), MAX_DATASETS),
        "training_setup": _clean_list(data.get("training_setup", []), MAX_TRAINING_SETUP),
        "results": _filter_generic_results(_clean_list(data.get("results", []), MAX_RESULTS)),
        "result_records": _clean_result_records(data.get("result_records", [])),
        "conclusions": _clean_list(data.get("conclusions", []), MAX_CONCLUSIONS),
        "limitations": _clean_list(data.get("limitations", []), MAX_LIMITATIONS),
        "future_work": _clean_list(data.get("future_work", []), MAX_FUTURE_WORK),
        "strengths": _clean_list(data.get("strengths", []), MAX_STRENGTHS),
        "claimed_novelty": _clean_list(data.get("claimed_novelty", []), MAX_NOVELTY),
        "entities": entities,
    }

    if not normalized["title"]:
        normalized["title"] = file_name.replace(".pdf", "").replace("_", " ").strip()

    if not normalized["datasets"] and entities["datasets"]:
        normalized["datasets"] = entities["datasets"][:]

    if not normalized["paper_type"]:
        normalized["paper_type"] = _infer_paper_type(
            normalized["title"], normalized["proposed_method"], entities
        )

    if normalized["paper_type"] == "survey_paper":
        if not normalized["proposed_method"] or "taxonomy" not in normalized["proposed_method"].lower():
            normalized["proposed_method"] = (
                "Survey/review-style synthesis of prior methods, datasets, and evaluation trends"
            )

        normalized["architecture_changes"] = []
        if not normalized["training_setup"]:
            normalized["training_setup"] = []

        if not normalized["claimed_novelty"]:
            normalized["claimed_novelty"] = [
                "Provides a structured synthesis of prior work rather than introducing one single novel architecture"
            ]

    return normalized


def _build_analysis_prompt(context: str, file_name: str, paper_id: str) -> str:
    return f"""
You are an expert academic paper analysis system.

Your task is to extract a CLEAN, STRICT, VALID JSON object from the paper context below.

Critical rules:
- Use ONLY the provided paper context.
- Do NOT invent details.
- Output VALID JSON ONLY.
- Do NOT include markdown fences.
- Do NOT include explanations before or after the JSON.
- If a field is missing, use "" for strings and [] for lists.
- Always preserve:
  - "paper_id": "{paper_id}"
  - "file_name": "{file_name}"

Very important:
- Classify the paper type as one of:
  - "method_paper"
  - "survey_paper"
  - "unclear"
- If the paper is a survey/review/benchmark/taxonomy paper, do NOT describe it as if it proposes one specific novel model.
- The title must be the actual paper title if present, not a generic topic name.
- Keep entity lists short and relevant.
- Do not dump very long lists of models/methods from the entire field unless clearly central to the paper.
- In "results", prefer dataset + metric + value statements.
- Do not output vague results like "good performance" or "high anomalous scores".
- "training_setup" should contain actual setup/protocol details only if they appear clearly.
- "architecture_changes" should be non-empty only for genuine method papers with clear architectural novelty.
- Extract structured performance records when possible.
- Identify the key strengths of the paper.
- Identify the claimed novelty or main contribution.
- If numerical results exist, capture dataset + metric + value in result_records.
- Keep result_records concise and only include clearly supported records from the context.

Return EXACTLY this JSON schema:

{{
  "paper_id": "{paper_id}",
  "file_name": "{file_name}",
  "paper_type": "",
  "title": "",
  "problem_statement": "",
  "proposed_method": "",
  "architecture_changes": [],
  "datasets": [],
  "training_setup": [],
  "results": [],
  "result_records": [
    {{
      "dataset": "",
      "metric": "",
      "value": "",
      "setting": "",
      "note": ""
    }}
  ],
  "conclusions": [],
  "limitations": [],
  "future_work": [],
  "strengths": [],
  "claimed_novelty": [],
  "entities": {{
    "models": [],
    "datasets": [],
    "methods": [],
    "tasks": [],
    "metrics": []
  }}
}}

Paper context:
{context}
""".strip()


def analyze_paper_context(context: str, file_name: str, paper_id: str) -> PaperAnalysis:
    if not context or not context.strip():
        raise ValueError("Paper context is empty.")

    llm = _get_llm()
    response = llm.invoke(_build_analysis_prompt(context, file_name, paper_id))
    raw_text = getattr(response, "content", str(response)).strip()

    json_text = _extract_json_object(raw_text)
    data = json.loads(json_text)
    normalized_data = _normalize_analysis_data(data, file_name=file_name, paper_id=paper_id)

    try:
        return PaperAnalysis.model_validate(normalized_data)
    except AttributeError:
        return PaperAnalysis.parse_obj(normalized_data)