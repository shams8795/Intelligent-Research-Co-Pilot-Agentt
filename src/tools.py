"""Tool definitions for the research agent.

This module is intentionally extensible: each new workflow can be
added as a new tool without changing the UI layer.
"""

import json

from langchain_core.tools import StructuredTool

from src.analysis_agent import analyze_paper_context
from src.knowledge_graph_agent import build_knowledge_graph
from src.summarizer import (
    build_retrieved_context_for_pdf,
    summarize_multiple_pdfs,
    summarize_pdf,
)


def get_agent_tools(uploaded_files=None):
    """Return a list of LangChain tools bound to the current uploaded PDFs."""
    files = uploaded_files or []

    def _summarize_paper() -> str:
        """Generate a detailed technical summary of the uploaded research paper(s)."""
        if not files:
            return "No PDF is uploaded. Please upload one or more research papers first."

        if len(files) == 1:
            return summarize_pdf(files[0])

        summaries = summarize_multiple_pdfs(files)
        return json.dumps(summaries, indent=2, ensure_ascii=False)

    def _paper_analysis_agent() -> str:
        """Generate structured analysis for each uploaded paper."""
        if not files:
            return "No PDF is uploaded. Please upload one or more research papers first."

        analyses = []
        for index, uploaded_file in enumerate(files, start=1):
            context = build_retrieved_context_for_pdf(uploaded_file)
            analysis = analyze_paper_context(
                context=context,
                file_name=uploaded_file.name,
                paper_id=f"paper_{index}",
            )
            analyses.append(analysis.model_dump())

        return json.dumps(analyses, indent=2, ensure_ascii=False)

    def _knowledge_graph_agent() -> str:
        """Build a cross-paper knowledge graph from uploaded papers."""
        if not files:
            return "No PDF is uploaded. Please upload one or more research papers first."

        analyses = []
        for index, uploaded_file in enumerate(files, start=1):
            context = build_retrieved_context_for_pdf(uploaded_file)
            analysis = analyze_paper_context(
                context=context,
                file_name=uploaded_file.name,
                paper_id=f"paper_{index}",
            )
            analyses.append(analysis)

        graph = build_knowledge_graph(analyses)
        return graph.model_dump_json(indent=2)

    summarize_tool = StructuredTool.from_function(
        func=_summarize_paper,
        name="summarize_paper",
        description=(
            "Generate a detailed summary of the uploaded research paper or papers. "
            "Use this when the user asks to summarize, get an overview, or describe the uploaded papers."
        ),
    )

    paper_analysis_tool = StructuredTool.from_function(
        func=_paper_analysis_agent,
        name="paper_analysis_agent",
        description=(
            "Generate structured analysis for each uploaded paper, including problem, method, datasets, "
            "results, limitations, future work, and extracted entities."
        ),
    )

    knowledge_graph_tool = StructuredTool.from_function(
        func=_knowledge_graph_agent,
        name="knowledge_graph_agent",
        description=(
            "Build a cross-paper knowledge graph from the uploaded papers, including nodes, edges, and insights."
        ),
    )

    return [
        summarize_tool,
        paper_analysis_tool,
        knowledge_graph_tool,
    ]