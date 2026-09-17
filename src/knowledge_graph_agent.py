from collections import defaultdict
from typing import Dict, List, Set

from src.schemas import (
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    PaperAnalysis,
    PaperComparisonEntry,
    ComparativeInsight,
    BestPaperCandidate,
)


MAX_SHARED_PER_TYPE = 8


def _normalize_id(value: str) -> str:
    cleaned = (value or "").strip().lower()
    cleaned = " ".join(cleaned.split())

    replacements = {
        "/": "_",
        "\\": "_",
        ":": "_",
        ",": "_",
        "(": "",
        ")": "",
        "[": "",
        "]": "",
        "{": "",
        "}": "",
        "'": "",
        '"': "",
        ".": "_",
        "-": "_",
        "%": "pct",
    }

    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)

    return cleaned.strip("_")


def _priority_entities(paper: PaperAnalysis) -> Dict[str, List[str]]:
    return {
        "datasets": (paper.entities.datasets or paper.datasets)[:6],
        "models": paper.entities.models[:5],
        "methods": paper.entities.methods[:5],
        "tasks": paper.entities.tasks[:3],
        "metrics": paper.entities.metrics[:4],
    }


# =========================
# 🔥 NEW: SHARED ELEMENTS
# =========================
def _compute_shared_elements(analyses: List[PaperAnalysis]) -> Dict[str, List[str]]:
    shared = {
        "datasets": defaultdict(list),
        "methods": defaultdict(list),
        "tasks": defaultdict(list),
    }

    for paper in analyses:
        name = paper.title or paper.file_name
        entities = _priority_entities(paper)

        for d in entities["datasets"]:
            shared["datasets"][d].append(name)

        for m in entities["methods"]:
            shared["methods"][m].append(name)

        for t in entities["tasks"]:
            shared["tasks"][t].append(name)

    # keep only shared ones
    return {
        key: {k: v for k, v in val.items() if len(v) > 1}
        for key, val in shared.items()
    }


# =========================
# 🔥 NEW: UNIQUE PER PAPER
# =========================
def _compute_unique_per_paper(analyses: List[PaperAnalysis]) -> List[PaperComparisonEntry]:
    all_methods = set()
    all_datasets = set()
    all_tasks = set()

    for paper in analyses:
        ent = _priority_entities(paper)
        all_methods.update(ent["methods"])
        all_datasets.update(ent["datasets"])
        all_tasks.update(ent["tasks"])

    result = []

    for paper in analyses:
        ent = _priority_entities(paper)

        entry = PaperComparisonEntry(
            paper_id=paper.paper_id,
            file_name=paper.file_name,
            title=paper.title,
            unique_methods=[m for m in ent["methods"] if sum(m in _priority_entities(p)["methods"] for p in analyses) == 1],
            unique_datasets=[d for d in ent["datasets"] if sum(d in _priority_entities(p)["datasets"] for p in analyses) == 1],
            unique_tasks=[t for t in ent["tasks"] if sum(t in _priority_entities(p)["tasks"] for p in analyses) == 1],
            unique_models=[],
            standout_points=paper.strengths[:3] if hasattr(paper, "strengths") else [],
        )

        result.append(entry)

    return result


# =========================
# 🔥 NEW: BEST PAPER LOGIC
# =========================
def _compute_best_papers(analyses: List[PaperAnalysis]) -> List[BestPaperCandidate]:
    candidates = []

    for paper in analyses:
        # Criterion 1: has structured results
        if getattr(paper, "result_records", []):
            candidates.append(
                BestPaperCandidate(
                    criterion="has structured quantitative results",
                    paper_id=paper.paper_id,
                    file_name=paper.file_name,
                    title=paper.title,
                    justification="This paper provides explicit dataset + metric + value results.",
                )
            )

        # Criterion 2: novelty
        if getattr(paper, "claimed_novelty", []):
            candidates.append(
                BestPaperCandidate(
                    criterion="novel contribution",
                    paper_id=paper.paper_id,
                    file_name=paper.file_name,
                    title=paper.title,
                    justification="This paper explicitly states novel contributions.",
                )
            )

        # Criterion 3: strong methods
        if paper.entities.methods:
            candidates.append(
                BestPaperCandidate(
                    criterion="method richness",
                    paper_id=paper.paper_id,
                    file_name=paper.file_name,
                    title=paper.title,
                    justification="This paper uses multiple or advanced methods.",
                )
            )

    return candidates[:5]


# =========================
# 🔥 NEW: COMPARATIVE INSIGHTS
# =========================
def _generate_comparative_insights(analyses: List[PaperAnalysis]) -> List[ComparativeInsight]:
    insights = []

    # common task
    task_count = defaultdict(int)
    for p in analyses:
        for t in p.entities.tasks:
            task_count[t] += 1

    for task, count in task_count.items():
        if count > 1:
            insights.append(
                ComparativeInsight(
                    category="shared_focus",
                    statement=f"Multiple papers focus on the task '{task}'.",
                    supporting_papers=[p.title for p in analyses if task in p.entities.tasks],
                )
            )

    # limitations trend
    all_limitations = []
    for p in analyses:
        all_limitations.extend(p.limitations)

    if len(all_limitations) > 2:
        insights.append(
            ComparativeInsight(
                category="limitations_trend",
                statement="Several papers report limitations such as limited evaluation or dataset constraints.",
                supporting_papers=[p.title for p in analyses],
            )
        )

    # novelty difference
    for p in analyses:
        if getattr(p, "claimed_novelty", []):
            insights.append(
                ComparativeInsight(
                    category="novelty",
                    statement=f"{p.title} introduces a distinct contribution compared to others.",
                    supporting_papers=[p.title],
                )
            )

    return insights


# =========================
# ORIGINAL GRAPH + NEW FEATURES
# =========================
def build_knowledge_graph(analyses: List[PaperAnalysis]) -> KnowledgeGraph:
    graph = KnowledgeGraph()
    seen_nodes: Set[str] = set()
    seen_edges: Set[str] = set()

    def add_node(node_id, node_type, label):
        if node_id in seen_nodes:
            return
        graph.nodes.append(GraphNode(id=node_id, type=node_type, label=label))
        seen_nodes.add(node_id)

    def add_edge(src, tgt, rel):
        key = f"{src}-{rel}-{tgt}"
        if key in seen_edges:
            return
        graph.edges.append(GraphEdge(source=src, target=tgt, relation=rel))
        seen_edges.add(key)

    for paper in analyses:
        pid = f"paper:{_normalize_id(paper.paper_id)}"
        add_node(pid, "paper", paper.title)

        ent = _priority_entities(paper)

        for d in ent["datasets"]:
            did = f"dataset:{_normalize_id(d)}"
            add_node(did, "dataset", d)
            add_edge(pid, did, "uses_dataset")

        for m in ent["methods"]:
            mid = f"method:{_normalize_id(m)}"
            add_node(mid, "method", m)
            add_edge(pid, mid, "uses_method")

    # 🔥 NEW PART
    graph.shared_elements = _compute_shared_elements(analyses)
    graph.per_paper_comparison = _compute_unique_per_paper(analyses)
    graph.comparative_insights = _generate_comparative_insights(analyses)
    graph.best_paper_candidates = _compute_best_papers(analyses)

    # keep old simple insights for UI
    graph.insights = [
        f"{len(analyses)} papers analyzed with cross-paper comparison enabled."
    ]

    return graph