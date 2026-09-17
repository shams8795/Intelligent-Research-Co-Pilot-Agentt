from typing import Any, Dict, List

from pydantic import BaseModel, Field


class PaperEntities(BaseModel):
    models: List[str] = Field(default_factory=list)
    datasets: List[str] = Field(default_factory=list)
    methods: List[str] = Field(default_factory=list)
    tasks: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)


class ResultRecord(BaseModel):
    dataset: str = ""
    metric: str = ""
    value: str = ""
    setting: str = ""
    note: str = ""


class PaperAnalysis(BaseModel):
    paper_id: str
    file_name: str
    paper_type: str = "unclear"

    title: str = ""
    problem_statement: str = ""
    proposed_method: str = ""

    architecture_changes: List[str] = Field(default_factory=list)
    datasets: List[str] = Field(default_factory=list)
    training_setup: List[str] = Field(default_factory=list)

    results: List[str] = Field(default_factory=list)
    result_records: List[ResultRecord] = Field(default_factory=list)

    conclusions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    future_work: List[str] = Field(default_factory=list)

    strengths: List[str] = Field(default_factory=list)
    claimed_novelty: List[str] = Field(default_factory=list)

    entities: PaperEntities = Field(default_factory=PaperEntities)


class GraphNode(BaseModel):
    id: str
    type: str
    label: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ComparativeInsight(BaseModel):
    category: str = ""
    statement: str = ""
    supporting_papers: List[str] = Field(default_factory=list)


class PaperComparisonEntry(BaseModel):
    paper_id: str
    file_name: str
    title: str = ""

    unique_methods: List[str] = Field(default_factory=list)
    unique_datasets: List[str] = Field(default_factory=list)
    unique_tasks: List[str] = Field(default_factory=list)
    unique_models: List[str] = Field(default_factory=list)

    standout_points: List[str] = Field(default_factory=list)


class BestPaperCandidate(BaseModel):
    criterion: str = ""
    paper_id: str = ""
    file_name: str = ""
    title: str = ""
    justification: str = ""


class KnowledgeGraph(BaseModel):
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)

    # Backward compatibility with current UI / old graph pipeline
    insights: List[str] = Field(default_factory=list)

    # New structured comparison outputs
    shared_elements: Dict[str, Dict[str, List[str]]] = Field(default_factory=dict)
    per_paper_comparison: List[PaperComparisonEntry] = Field(default_factory=list)
    comparative_insights: List[ComparativeInsight] = Field(default_factory=list)
    best_paper_candidates: List[BestPaperCandidate] = Field(default_factory=list)