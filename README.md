# Intelligent Research Co-Pilot

### AI-Powered Research Paper Analysis, RAG, Knowledge Graphs, Hypothesis Generation, and Multimodal Q&A

Intelligent Research Co-Pilot is an AI-powered research assistant designed to help researchers analyze, summarize, compare, question, and extract structured insights from academic research papers.

The application combines Retrieval-Augmented Generation (RAG), large language models, semantic retrieval, structured paper analysis, cross-paper comparison, knowledge graphs, hypothesis generation, and multimodal question answering inside an interactive Streamlit interface.

The system supports uploading up to 5 research papers at once and provides multiple research workflows grounded in the uploaded documents.

---

# Overview

Reading and comparing multiple academic papers can be time-consuming.

Researchers often need to:

- Understand the main research problem
- Identify the proposed method
- Extract datasets and evaluation metrics
- Compare methods across multiple papers
- Identify limitations and future research directions
- Ask detailed questions about paper content
- Analyze scientific figures
- Detect possible research gaps
- Generate new research hypotheses
- Plan experiments based on existing literature

Intelligent Research Co-Pilot brings these tasks together in a single research workspace.

The application processes uploaded PDF papers, retrieves relevant evidence, and uses AI models to generate grounded research insights.

---

# Main Capabilities

The application provides the following research workflows:

- Research paper summarization
- Structured paper analysis
- Multi-paper comparison
- Knowledge graph generation
- Paper-grounded question answering
- Scientific figure analysis
- Research gap detection
- Hypothesis generation
- Experiment planning
- AI agent-based research assistance

---

# Supported Research Papers

The application supports:

```text
1 to 5 PDF research papers per session
```

Users can upload multiple papers and move between different research workflows while keeping the uploaded documents available in the current Streamlit session.

---

# Research Paper Summarization

The project includes a Retrieval-Augmented Generation pipeline for generating technical paper summaries.

Instead of sending the entire PDF directly to the language model, the application:

```text
PDF
 ↓
Document Loading
 ↓
Text Chunking
 ↓
Embedding Generation
 ↓
Chroma Vector Store
 ↓
MMR Retrieval
 ↓
Relevant Context
 ↓
Groq LLM
 ↓
Technical Summary
```

The summarization pipeline retrieves relevant sections of the paper using several research-oriented queries.

These queries focus on:

- Research problem and objective
- Proposed technical method
- Core contribution
- Experimental results
- Evaluation metrics
- Real-world impact
- Limitations
- Future work

---

# Summary Output

The generated technical summary follows a structured format:

### Problem Statement

Explains the problem addressed by the paper and why existing methods may be insufficient.

### Proposed Solution

Describes the paper's main technical approach and contribution.

### Key Insight

Highlights the most important technical idea behind the proposed method.

### Important Results

Extracts important reported evaluation results and metrics when available.

### Real-World Impact

Explains the possible technical or practical impact of the research.

### Limitations

Identifies limitations supported by the retrieved paper context.

The summarization prompt explicitly instructs the model not to invent missing numerical results or unsupported metrics.

---

# Retrieval-Augmented Generation

The summarization workflow uses RAG to retrieve relevant sections before generating an answer.

The project uses:

- ChromaDB for vector storage
- Chroma local embeddings
- Maximum Marginal Relevance retrieval
- Multiple research-specific retrieval queries
- Context deduplication
- Page-aware context ordering

The default retriever uses MMR to improve diversity between retrieved chunks.

This helps reduce repeated or near-duplicate passages and improves coverage across different sections of a research paper.

---

# Document Chunking

Research papers are divided into smaller text chunks before embedding.

The project uses:

```text
RecursiveCharacterTextSplitter
```

The general splitting module uses:

```text
Chunk Size: 1200
Chunk Overlap: 75
```

For the summarization pipeline, the project uses:

```text
Chunk Size: 1500
Chunk Overlap: 100
```

This allows the system to preserve useful local context while keeping individual chunks manageable for retrieval.

---

# Vector Store

The main RAG summarization pipeline uses:

```text
ChromaDB
```

The vector store is created in memory for the uploaded documents.

The application uses Chroma's built-in embedding function through a LangChain embedding adapter.

This allows the project to create semantic vectors without requiring OpenAI embeddings.

---

# Paper Analysis Agent

The project includes a structured paper analysis pipeline.

Each paper can be analyzed into structured research information including:

- Paper type
- Paper title
- Problem statement
- Proposed method
- Architecture changes
- Datasets
- Training setup
- Results
- Numerical result records
- Conclusions
- Limitations
- Future work
- Strengths
- Claimed novelty
- Models
- Methods
- Tasks
- Metrics

The output is validated using Pydantic schemas.

---

# Paper Type Detection

The paper analysis pipeline can distinguish between different paper styles.

The project includes logic for identifying papers such as:

```text
method_paper
survey_paper
```

Survey papers are handled differently from method papers.

For example, the system avoids treating survey papers as if they introduced a new model architecture when the paper is primarily reviewing or organizing previous research.

---

# Structured Numerical Results

When numerical results are clearly available in the retrieved paper context, the project can store them in structured records.

Each result can include:

```text
Dataset
Metric
Value
Experimental Setting
Additional Note
```

This provides more structured information than a simple natural-language paper summary.

---

# Multi-Paper Analysis

When multiple research papers are uploaded, the project can analyze them together.

The multi-paper workflow:

```text
Upload Multiple PDFs
        ↓
Retrieve Relevant Context
        ↓
Generate Individual Summaries
        ↓
Generate Structured Paper Analysis
        ↓
Build Cross-Paper Knowledge Graph
        ↓
Generate Comparison Insights
```

The maximum supported number of uploaded papers is 5.

---

# Knowledge Graph Agent

The project can generate a knowledge graph across the uploaded papers.

The graph identifies connections between papers and research entities.

The current graph implementation creates nodes for:

- Papers
- Datasets
- Methods

And relationships such as:

```text
Paper → uses_dataset → Dataset
Paper → uses_method → Method
```

The project also extracts additional entities including:

- Models
- Tasks
- Metrics

These are used in cross-paper comparison logic.

---

# Shared Research Elements

The cross-paper analysis detects research elements shared by multiple papers.

The system currently compares shared:

- Datasets
- Methods
- Tasks

This makes it easier to identify where different papers overlap.

---

# Unique Elements Per Paper

The system also identifies elements that appear uniquely in individual papers.

Examples include:

- Unique methods
- Unique datasets
- Unique tasks
- Standout research points

This helps distinguish the contribution and focus of each paper.

---

# Comparative Insights

The project generates structured comparative insights across papers.

Current comparison logic can identify patterns such as:

- Shared research tasks
- Common limitations
- Differences in claimed novelty
- Papers containing explicit quantitative results
- Papers with method-rich contributions

---

# Paper Comparison Candidates

The knowledge graph pipeline also identifies papers that meet specific comparison criteria.

Examples include papers that:

- Contain structured quantitative results
- Explicitly claim novel contributions
- Use multiple or advanced methods

These candidates are presented as structured comparison information.

They should be interpreted as rule-based comparison indicators rather than a universal ranking of paper quality.

---

# Ask Questions from Paper

The project includes a paper-grounded question answering system.

Users can ask natural-language questions about the uploaded papers.

The Q&A pipeline uses:

```text
Uploaded PDFs
      ↓
Text Extraction
      ↓
Text Chunking
      ↓
Sentence Transformer Embeddings
      ↓
FAISS Vector Search
      ↓
Relevant Text Evidence
      ↓
Groq Reasoning
      ↓
Grounded Answer
```

The Q&A system uses:

```text
all-MiniLM-L6-v2
```

for semantic text embeddings.

FAISS is used for text retrieval.

---

# Multimodal Research Q&A

The question answering pipeline also extracts images and figures from uploaded research papers.

The project uses:

```text
PyMuPDF
```

to extract:

- PDF text
- Embedded images
- Scientific figures

The system then uses:

```text
clip-ViT-B-32
```

to rank extracted figures according to their semantic relevance to the user's question.

The most relevant figures can then be passed to a vision-capable Groq model for analysis.

---

# Scientific Figure Analysis

Relevant research figures can be analyzed by the AI system.

The image analysis prompt asks the model to:

```text
Analyze the scientific plot or figure and explain the key insights.
```

The resulting image analysis is combined with retrieved text evidence before generating the final research answer.

This allows the Q&A system to reason over both:

- Text evidence
- Image evidence

---

# Q&A Confidence Value

The Q&A workflow returns a calculated confidence value based on retrieval distance.

The returned answer contains:

```text
AI-generated answer

Confidence: value
```

This value is derived programmatically from the FAISS retrieval distances.

It should be interpreted as a retrieval-based heuristic rather than a calibrated probability of answer correctness.

---

# AI Research Agent

The project includes a LangGraph ReAct agent for research-oriented workflows.

The agent uses:

```text
LangGraph
LangChain
ChatGroq
```

The default agent model is:

```text
llama-3.3-70b-versatile
```

unless another model is configured through the environment.

The research agent is instructed to ground responses in uploaded paper content.

---

# Agent Tools

The AI research agent can access tools bound to the currently uploaded research papers.

The implemented tools include workflows for:

### Paper Summarization

Generates technical summaries for one or multiple uploaded papers.

### Paper Analysis

Generates structured analysis for each paper.

### Knowledge Graph Generation

Builds cross-paper knowledge graph information from structured paper analyses.

This tool architecture makes the research agent extensible.

Additional tools can be added without requiring major changes to the Streamlit user interface.

---

# Research Gap Detection

The user interface includes a Research Gap Detection workspace.

The workflow sends a research-oriented instruction to the LangGraph agent:

```text
Detect the main research gaps across the uploaded papers.
```

The result is generated using the agent and its available paper-analysis tools.

---

# Hypothesis Generation

The project contains a dedicated hypothesis generation module.

The hypothesis pipeline processes uploaded papers and extracts structured information such as:

- Research problems
- Methods
- Datasets
- Limitations
- Future work

It then performs semantic gap detection across the papers.

---

# Research Gap Types

The hypothesis-generation pipeline includes logic for detecting research gaps such as:

### Limitation Gaps

Generated when limitations across papers suggest an underexplored research problem.

### Method Gaps

Can identify possible differences or missing methodological directions.

### Dataset Gaps

Can identify underexplored dataset-related directions across papers.

---

# Hypothesis Generation Pipeline

The hypothesis workflow follows approximately this process:

```text
Uploaded Papers
      ↓
PDF Text Extraction
      ↓
Structured Paper Extraction
      ↓
Cross-Paper Signal Extraction
      ↓
Semantic Gap Detection
      ↓
Hypothesis Generation
      ↓
Hypothesis Scoring
      ↓
Ranked Hypotheses
```

Generated hypotheses can include fields such as:

- Hypothesis
- Reasoning
- Novelty rationale
- Gap type
- Score

---

# Hypothesis Scoring

The project calculates a score for generated hypotheses.

The score considers multiple factors including:

- Reasoning quality
- Semantic novelty
- Evidence grounding

Sentence Transformer embeddings are used as part of the semantic novelty calculation.

The generated hypotheses are sorted by score.

---

# Experiment Planning

The application includes an Experiment Planning workspace.

The user interface sends the following type of research request to the AI agent:

```text
Plan a suitable experiment based on the uploaded papers.
```

The output is produced by the general research agent using the available uploaded-paper context and tools.

---

# Groq Integration

Groq is used as the main cloud LLM provider in several parts of the project.

The default research model used in major workflows is:

```text
llama-3.3-70b-versatile
```

The model can be changed using the:

```text
GROQ_MODEL
```

environment variable.

A:

```text
GROQ_API_KEY
```

environment variable is required for Groq-powered functionality.

---

# Ollama Fallback

The summarization pipeline includes an Ollama fallback for Groq API rate-limit situations.

If the Groq API returns a rate-limit error and a local Ollama server is available, the summarizer can use:

```text
Mistral
```

locally.

This allows summarization to continue without depending entirely on the cloud API.

---

# Ollama Setup

The repository includes:

```text
OLLAMA_SETUP.md
```

with instructions for installing and running Ollama locally.

The local setup uses:

```bash
ollama pull mistral
```

and:

```bash
ollama serve
```

The Ollama server runs locally at:

```text
http://localhost:11434
```

---

# Application Interface

The project uses Streamlit to provide an interactive research workspace.

The interface includes dedicated feature cards for:

- Summarize Paper
- Paper Analysis Agent
- Knowledge Graph Agent
- Ask Questions from Paper
- Research Gap Detection
- Hypothesis Generation
- Experiment Planning

The interface allows users to upload papers once and then move between different research workflows.

---

# Technology Stack

The project uses:

### Python

Core programming language.

### Streamlit

Interactive research application interface.

### LangChain

LLM orchestration, document workflows, tools, and retrieval integration.

### LangGraph

ReAct research agent implementation.

### Groq

Cloud-based large language model inference.

### ChromaDB

Vector storage for the main RAG summarization workflow.

### FAISS

Semantic vector search for the Q&A workflow.

### Sentence Transformers

Text and multimodal embedding generation.

### PyPDF

PDF document loading.

### PyMuPDF

PDF text and image extraction for multimodal Q&A.

### CLIP

Semantic image retrieval for scientific figures.

### Pydantic

Structured research data validation.

### NumPy

Vector and similarity operations.

### Pillow

Image processing.

### Ollama

Optional local LLM fallback for summarization.

---

# Python Dependencies

The included requirements file contains:

```text
streamlit
langchain
langchain-core
langchain-openai
langchain-groq
langchain-community
langgraph
chromadb
pypdf
pymupdf
python-dotenv
tiktoken
sentence-transformers
faiss-cpu
numpy
pillow
groq
torch
transformers
langchain-text-splitters
```

---

# Project Structure

```text
Intelligent-Research-Co-Pilot-Agent/
│
├── app.py
├── config.py
├── requirements.txt
├── OLLAMA_SETUP.md
│
├── data/
│   └── papers/
│
└── src/
    ├── __init__.py
    ├── agent.py
    ├── analysis_agent.py
    ├── hypothesis_agent_v2.py
    ├── knowledge_graph_agent.py
    ├── multi_paper_service.py
    ├── ollama_handler.py
    ├── pdf_loader.py
    ├── qa_card_agent.py
    ├── retriever.py
    ├── schemas.py
    ├── summarizer.py
    ├── text_splitter.py
    ├── tools.py
    └── vector_store.py
```

---

# Main Modules

## app.py

Contains the Streamlit user interface and coordinates the different research workflows.

## agent.py

Implements the LangGraph ReAct research agent using ChatGroq.

## summarizer.py

Implements the RAG-based research paper summarization workflow.

## analysis_agent.py

Extracts structured research information from retrieved paper context.

## knowledge_graph_agent.py

Builds cross-paper knowledge graphs and comparison information.

## hypothesis_agent_v2.py

Detects research gaps and generates ranked research hypotheses.

## qa_card_agent.py

Implements multimodal paper-grounded Q&A using FAISS, Sentence Transformers, CLIP, and Groq.

## multi_paper_service.py

Coordinates multi-paper summarization, structured analysis, and knowledge graph creation.

## pdf_loader.py

Loads uploaded PDFs using PyPDFLoader.

## text_splitter.py

Splits loaded documents into retrieval-friendly chunks.

## vector_store.py

Creates the Chroma vector store and local embedding adapter.

## retriever.py

Creates the MMR-based retriever used by the RAG pipeline.

## tools.py

Defines LangChain tools available to the research agent.

## schemas.py

Contains Pydantic data models for paper analysis, knowledge graphs, result records, and comparison outputs.

## ollama_handler.py

Handles local Ollama availability and local summarization fallback.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Intelligent-Research-Co-Pilot-Agent.git
```

## 2. Open the Project Folder

```bash
cd Intelligent-Research-Co-Pilot-Agent
```

## 3. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Configuration

Create a local file named:

```text
.env
```

Add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Optional model configuration:

```env
GROQ_MODEL=llama-3.3-70b-versatile
```

Do not upload your real `.env` file to GitHub.

---

# Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application is normally available at:

```text
http://localhost:8501
```

---

# Optional Local Ollama Setup

For local summarization fallback, install Ollama and download Mistral:

```bash
ollama pull mistral
```

Start the Ollama server:

```bash
ollama serve
```

Then run the Streamlit application normally.

---

# Security

Do not commit API keys or private credentials to the repository.

Your `.gitignore` should include:

```gitignore
.env
__pycache__/
*.pyc
.venv/
```

An example environment file can be included instead:

```text
.env.example
```

Example:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

---

# Current Capabilities

The current project includes working code for:

- Uploading up to 5 PDF research papers
- RAG-based paper summarization
- Semantic paper retrieval
- Chroma vector storage
- MMR retrieval
- Structured paper analysis
- Multi-paper analysis
- Cross-paper knowledge graph generation
- Shared and unique research element detection
- Comparative research insights
- Paper-grounded text Q&A
- Scientific figure extraction
- CLIP-based image relevance ranking
- Multimodal Q&A
- Research gap workflow
- Dedicated hypothesis generation
- Hypothesis scoring
- Experiment planning workspace
- LangGraph ReAct agent
- Groq LLM integration
- Ollama summarization fallback

---

# Future Improvements

Possible future improvements include:

- Persistent vector databases
- Citation generation with exact page references
- More robust PDF table extraction
- Stronger figure and chart understanding
- Research paper metadata extraction
- DOI and bibliographic integration
- Exportable literature review reports
- Research project history
- User authentication
- Persistent research workspaces
- Additional knowledge graph entity types
- More advanced hypothesis validation
- Experiment tracking
- Academic search engine integration

---

# About the Project

Intelligent Research Co-Pilot was developed as an AI research assistance system for working with academic literature.

The project demonstrates how Retrieval-Augmented Generation, LLM agents, structured AI extraction, vector databases, multimodal embeddings, knowledge graphs, and research-oriented reasoning workflows can be combined into a single application.

The goal is to reduce repetitive literature-analysis work while keeping AI outputs grounded in uploaded academic papers.

---

<h2 align="center">Intelligent Research Co-Pilot</h2>

<p align="center">
AI-Powered Research Paper Analysis, RAG, Knowledge Graphs, Hypothesis Generation, and Multimodal Q&A
</p>
