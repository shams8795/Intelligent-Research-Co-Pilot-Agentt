from typing import Any, Dict

from src.analysis_agent import analyze_paper_context
from src.knowledge_graph_agent import build_knowledge_graph
from src.summarizer import build_retrieved_context_for_pdf, summarize_pdf


def _dump_model(obj):
    try:
        return obj.model_dump()
    except AttributeError:
        return obj.dict()


def analyze_multiple_papers(uploaded_files) -> Dict[str, Any]:
    if not uploaded_files:
        raise ValueError("Please upload at least one PDF.")

    if len(uploaded_files) > 5:
        raise ValueError("Maximum supported PDFs is 5.")

    summaries: Dict[str, str] = {}
    analyses = []

    for index, uploaded_file in enumerate(uploaded_files):
        paper_id = f"paper_{index}"

        context = build_retrieved_context_for_pdf(uploaded_file)
        summary = summarize_pdf(uploaded_file)
        analysis = analyze_paper_context(
            context=context,
            file_name=uploaded_file.name,
            paper_id=paper_id,
        )

        summaries[uploaded_file.name] = summary
        analyses.append(analysis)

    graph = build_knowledge_graph(analyses)

    return {
        "summaries": summaries,
        "paper_analyses": [_dump_model(analysis) for analysis in analyses],
        "knowledge_graph": _dump_model(graph),
    }