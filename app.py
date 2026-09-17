"""Streamlit UI for the Intelligent Research Co-Pilot project.

Expected backend contract:
- src.summarizer.summarize_pdf(uploaded_file) -> str
- src.summarizer.summarize_multiple_pdfs(uploaded_files) -> dict
- src.agent.run_agent(user_request, uploaded_files=None) -> str
"""

import html

import streamlit as st

from src import agent as agent_backend
from src import qa_card_agent as qa_card_backend
from src import summarizer as summarizer_backend
from src.multi_paper_service import analyze_multiple_papers
from src.hypothesis_agent_v2 import generate_hypotheses_v2

summarize_pdf = getattr(summarizer_backend, "summarize_pdf", None)
summarize_multiple_pdfs = getattr(summarizer_backend, "summarize_multiple_pdfs", None)
run_agent = getattr(agent_backend, "run_agent", None)
qa_answer_question = getattr(qa_card_backend, "answer_question", None)

FEATURES = [
    {
        "key": "summarize_paper",
        "eyebrow": "Synthesis",
        "title": "Summarize Paper",
        "description": "Generate a concise summary once the paper is uploaded and processed by the backend.",
        "button": "Open Summary",
    },
    {
        "key": "paper_analysis",
        "eyebrow": "Structure",
        "title": "Paper Analysis Agent",
        "description": "Generate structured analysis for each uploaded paper.",
        "button": "Open Analysis",
    },
    {
        "key": "knowledge_graph",
        "eyebrow": "Connections",
        "title": "Knowledge Graph Agent",
        "description": "Build a cross-paper knowledge graph and detect links across the uploaded papers.",
        "button": "Open Graph",
    },
    {
        "key": "ask_questions",
        "eyebrow": "Reasoning",
        "title": "Ask Questions from Paper",
        "description": "Send questions to the agent and retrieve answers grounded in the uploaded paper(s).",
        "button": "Open Q&A",
    },
    {
        "key": "research_gap_detection",
        "eyebrow": "Discovery",
        "title": "Research Gap Detection",
        "description": "Ask the AI agent to identify underexplored directions and missing comparisons.",
        "button": "Open Gaps",
        "agent_prompt": "Detect the main research gaps across the uploaded papers.",
    },
    {
        "key": "hypothesis_generation",
        "eyebrow": "Ideation",
        "title": "Hypothesis Generation",
        "description": "Use the agent to turn literature findings into clear, testable hypotheses.",
        "button": "Open Hypotheses",
        "agent_prompt": "Generate strong research hypotheses based on the uploaded papers.",
    },
    {
        "key": "experiment_planning",
        "eyebrow": "Execution",
        "title": "Experiment Planning",
        "description": "Create a structured experiment plan driven by the uploaded paper(s) and the agent.",
        "button": "Open Planner",
        "agent_prompt": "Plan a suitable experiment based on the uploaded papers.",
    },
]


def configure_page() -> None:
    st.set_page_config(
        page_title="Intelligent Research Co-Pilot",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def initialize_session_state() -> None:
    defaults = {
        "active_feature": None,
        "uploaded_file": None,
        "uploaded_file_name": None,
        "summary_text": "",
        "summaries_dict": {},
        "paper_analyses": [],
        "knowledge_graph": {},
        "feature_outputs": {},
        "chat_messages": [
            {
                "role": "assistant",
                "content": (
                    "Hello. I am your Intelligent Research Co-Pilot. Upload papers (1-5), then ask "
                    "questions or open a workflow below."
                ),
            }
        ],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Manrope:wght@400;500;600;700;800&display=swap');

        :root {
            --page-bg: #f3efe7;
            --card-bg: rgba(255, 255, 255, 0.88);
            --border-soft: rgba(24, 53, 77, 0.12);
            --text-strong: #13324a;
            --text-soft: #5a6d7e;
            --accent: #0e7c86;
            --accent-dark: #0a5960;
            --shadow: 0 18px 45px rgba(19, 50, 74, 0.12);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(195, 155, 93, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(14, 124, 134, 0.12), transparent 24%),
                linear-gradient(180deg, var(--page-bg) 0%, #f9f7f2 46%, #edf3f5 100%);
            color: var(--text-strong);
            font-family: 'Manrope', 'Trebuchet MS', sans-serif;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3 {
            font-family: 'Fraunces', Georgia, serif;
            color: var(--text-strong);
            letter-spacing: -0.02em;
        }

        h4 {
            color: var(--text-strong);
            margin-bottom: 0.4rem;
        }

        p, li, label, span {
            font-family: 'Manrope', 'Trebuchet MS', sans-serif;
        }

        .section-header {
            margin: 0 0 1.4rem 0;
        }

        .section-kicker {
            display: inline-block;
            padding: 0.32rem 0.7rem;
            border-radius: 999px;
            background: rgba(14, 124, 134, 0.09);
            border: 1px solid rgba(14, 124, 134, 0.12);
            color: var(--accent-dark);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.9rem;
        }

        .section-header h2 {
            margin: 0;
            font-size: 2rem;
        }

        .section-header p {
            margin: 0.55rem 0 0 0;
            max-width: 760px;
            color: var(--text-soft);
            font-size: 1rem;
            line-height: 1.7;
        }

        .hero-copy,
        .assistant-visual,
        .feature-card,
        .workspace-card,
        .summary-box,
        .placeholder-box,
        .chat-teaser,
        .chat-shell,
        .message-card,
        .backend-note,
        .analysis-card,
        .comparison-card {
            background: var(--card-bg);
            border: 1px solid var(--border-soft);
            border-radius: 28px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(16px);
        }

        .hero-copy {
            padding: 2rem 2rem 2.1rem 2rem;
            min-height: 100%;
        }

        .hero-copy h1 {
            margin: 0;
            font-size: 3.45rem;
            line-height: 1.03;
        }

        .hero-copy p {
            margin: 1rem 0 0 0;
            color: var(--text-soft);
            font-size: 1.06rem;
            line-height: 1.8;
            max-width: 640px;
        }

        .hero-stats {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1.5rem;
        }

        .hero-stat {
            padding: 1rem 1.05rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.74);
            border: 1px solid rgba(19, 50, 74, 0.08);
        }

        .hero-stat strong {
            display: block;
            font-size: 0.98rem;
            color: var(--text-strong);
        }

        .hero-stat span {
            display: block;
            margin-top: 0.3rem;
            color: var(--text-soft);
            font-size: 0.84rem;
            line-height: 1.55;
        }

        .assistant-visual {
            padding: 1.55rem;
            position: relative;
            overflow: hidden;
            min-height: 100%;
            animation: floatCard 5s ease-in-out infinite;
        }

        .assistant-visual::after {
            content: "";
            position: absolute;
            inset: auto -40px -40px auto;
            width: 180px;
            height: 180px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(14, 124, 134, 0.18), transparent 70%);
        }

        .assistant-status {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            color: var(--accent-dark);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            margin-bottom: 1rem;
        }

        .assistant-status-dot {
            width: 0.7rem;
            height: 0.7rem;
            border-radius: 50%;
            background: var(--accent);
            box-shadow: 0 0 0 0 rgba(14, 124, 134, 0.35);
            animation: pulseDot 1.8s infinite;
        }

        .assistant-window {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(19, 50, 74, 0.08);
            border-radius: 22px;
            padding: 1.1rem;
        }

        .visual-bubble {
            max-width: 88%;
            padding: 0.85rem 1rem;
            border-radius: 18px;
            font-size: 0.92rem;
            line-height: 1.55;
            margin-bottom: 0.85rem;
        }

        .visual-bubble.user {
            background: #e7f1f4;
            margin-left: auto;
            color: var(--text-strong);
        }

        .visual-bubble.assistant {
            background: #f8f4ea;
            color: var(--text-strong);
        }

        .assistant-orb {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 78px;
            height: 78px;
            border-radius: 50%;
            margin-top: 1.05rem;
            background: linear-gradient(145deg, rgba(14, 124, 134, 0.15), rgba(195, 155, 93, 0.22));
            border: 1px solid rgba(14, 124, 134, 0.15);
            font-family: 'Fraunces', Georgia, serif;
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--accent-dark);
        }

        .chat-teaser,
        .chat-shell,
        .workspace-card,
        .summary-box,
        .placeholder-box,
        .backend-note,
        .analysis-card,
        .comparison-card {
            padding: 1.45rem 1.5rem;
        }

        .chat-teaser h3,
        .chat-shell h3,
        .workspace-card h3,
        .summary-box h3,
        .placeholder-box h3,
        .feature-card h3,
        .backend-note h3,
        .analysis-card h3,
        .comparison-card h3 {
            margin: 0;
            font-size: 1.35rem;
        }

        .chat-teaser p,
        .chat-shell p,
        .workspace-card p,
        .summary-box p,
        .placeholder-box p,
        .feature-card p,
        .backend-note p {
            color: var(--text-soft);
            line-height: 1.7;
        }

        .chat-badge,
        .feature-eyebrow,
        .panel-eyebrow,
        .summary-label {
            display: inline-block;
            color: var(--accent-dark);
            font-weight: 800;
            font-size: 0.74rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.7rem;
        }

        .feature-card {
            padding: 1.3rem 1.3rem 1.4rem 1.3rem;
            min-height: 240px;
            height: 240px;
            display: flex;
            flex-direction: column;
            transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
            margin-bottom: 0.9rem;
        }

        .feature-card p {
            margin-top: 0.75rem;
            margin-bottom: 0;
        }

        .feature-card:hover,
        .chat-teaser:hover,
        .workspace-card:hover,
        .summary-box:hover,
        .placeholder-box:hover,
        .backend-note:hover,
        .analysis-card:hover,
        .comparison-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 24px 50px rgba(19, 50, 74, 0.16);
            border-color: rgba(14, 124, 134, 0.18);
        }

        .file-pill {
            display: inline-block;
            margin-top: 0.9rem;
            padding: 0.65rem 0.95rem;
            border-radius: 999px;
            background: rgba(14, 124, 134, 0.08);
            color: var(--accent-dark);
            border: 1px solid rgba(14, 124, 134, 0.12);
            font-size: 0.9rem;
            font-weight: 700;
        }

        .message-card {
            padding: 0.95rem 1rem;
            margin-bottom: 0.8rem;
        }

        .message-card.user {
            background: rgba(231, 241, 244, 0.92);
        }

        .message-card.assistant {
            background: rgba(248, 244, 234, 0.96);
        }

        .message-role {
            color: var(--accent-dark);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .message-text {
            color: var(--text-strong);
            line-height: 1.65;
            font-size: 0.95rem;
        }

        .mini-tag {
            display: inline-block;
            margin: 0.2rem 0.35rem 0.2rem 0;
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            background: rgba(14, 124, 134, 0.08);
            border: 1px solid rgba(14, 124, 134, 0.12);
            color: var(--accent-dark);
            font-size: 0.8rem;
            font-weight: 700;
        }

        .subtle-text {
            color: var(--text-soft);
            line-height: 1.7;
        }

        div[data-testid="stFileUploader"] {
            border-radius: 22px;
            border: 1px dashed rgba(14, 124, 134, 0.32);
            background: rgba(255, 255, 255, 0.72);
            padding: 0.4rem;
        }

        div[data-testid="stTextInputRootElement"] input {
            border-radius: 16px;
            border: 1px solid rgba(19, 50, 74, 0.12);
            background: rgba(255, 255, 255, 0.84);
        }

        .stButton > button,
        .stFormSubmitButton > button {
            width: 100%;
            min-height: 2.85rem;
            border-radius: 16px;
            border: 0;
            background: linear-gradient(135deg, var(--accent) 0%, #155f73 100%);
            color: #ffffff;
            font-weight: 700;
            letter-spacing: 0.01em;
            box-shadow: 0 10px 24px rgba(14, 124, 134, 0.25);
            transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            transform: translateY(-2px);
            filter: brightness(1.03);
            box-shadow: 0 14px 28px rgba(14, 124, 134, 0.3);
        }

        @keyframes pulseDot {
            0% {
                box-shadow: 0 0 0 0 rgba(14, 124, 134, 0.35);
            }
            70% {
                box-shadow: 0 0 0 12px rgba(14, 124, 134, 0);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(14, 124, 134, 0);
            }
        }

        @keyframes floatCard {
            0% {
                transform: translateY(0px);
            }
            50% {
                transform: translateY(-6px);
            }
            100% {
                transform: translateY(0px);
            }
        }

        @media (max-width: 900px) {
            .hero-copy h1 {
                font-size: 2.75rem;
            }

            .hero-stats {
                grid-template-columns: 1fr;
            }

            .section-header h2 {
                font-size: 1.7rem;
            }

            .feature-card {
                height: auto;
                min-height: 220px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_html_text(text: str) -> str:
    return html.escape(str(text)).replace("\n", "<br>")


def render_section_header(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-kicker">{kicker}</div>
            <h2>{title}</h2>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def set_active_feature(feature_key: str) -> None:
    st.session_state.active_feature = feature_key


def sync_uploaded_file(uploaded_files) -> None:
    current_names = [f.name for f in uploaded_files] if uploaded_files else []
    previous_names = st.session_state.uploaded_file_name or []

    st.session_state.uploaded_file = uploaded_files
    st.session_state.uploaded_file_name = current_names

    if current_names != previous_names:
        st.session_state.summary_text = ""
        st.session_state.summaries_dict = {}
        st.session_state.paper_analyses = []
        st.session_state.knowledge_graph = {}
        st.session_state.feature_outputs = {}


def missing_backend_message(function_name: str, module_path: str) -> str:
    return (
        f"Backend hook '{function_name}' is not available yet. Add it to {module_path} "
        "and the UI will start using it automatically."
    )


def get_active_paper_title() -> str:
    if not st.session_state.uploaded_file_name:
        return "Paper synopsis"

    if isinstance(st.session_state.uploaded_file_name, list):
        if len(st.session_state.uploaded_file_name) == 1:
            return f"{st.session_state.uploaded_file_name[0]} Summary"
        return "Multi-Paper Summary"

    return f"{st.session_state.uploaded_file_name} Summary"


def request_summary() -> None:
    uploaded_files = st.session_state.uploaded_file
    if not uploaded_files:
        st.session_state.summary_text = "Upload at least one research paper PDF before generating summaries."
        return

    summarize_func = None
    if len(uploaded_files) == 1 and summarize_pdf is not None:
        summarize_func = summarize_pdf
    elif len(uploaded_files) > 1:
        summarize_func = getattr(summarizer_backend, "summarize_multiple_pdfs", None)
    else:
        summarize_func = summarize_pdf

    if summarize_func is None:
        st.session_state.summary_text = missing_backend_message(
            "summarize_pdf/summarize_multiple_pdfs", "src/summarizer.py"
        )
        return

    try:
        with st.spinner(f"Generating summaries for {len(uploaded_files)} file(s)..."):
            if len(uploaded_files) == 1:
                generated_summary = str(summarize_pdf(uploaded_files[0])).strip()
                st.session_state.summary_text = generated_summary
                st.session_state.summaries_dict = {uploaded_files[0].name: generated_summary}
            else:
                summaries_dict = summarize_multiple_pdfs(uploaded_files)
                st.session_state.summaries_dict = summaries_dict
                st.session_state.summary_text = "Multiple summaries generated. See below for details."

        if not st.session_state.summary_text:
            st.session_state.summary_text = (
                "Summary generation returned an empty response. "
                "Please try again or upload clearer PDF files."
            )
    except Exception as error:
        error_str = str(error)
        if "Rate limit" in error_str or "429" in error_str or "tokens per day" in error_str.lower():
            st.session_state.summary_text = (
                "⚠️ API Rate Limit Reached\n\n"
                "The daily token limit for the summarization API has been reached. "
                "This will reset tomorrow (in ~24 hours). Please try again after the limit resets.\n\n"
                "To avoid this in the future, consider:\n"
                "- Using a different API key with higher limits\n"
                "- Upgrading to a higher tier plan\n"
                "- Waiting for the daily reset"
            )
        else:
            st.session_state.summary_text = f"Summary generation failed: {error}"

def request_hypotheses():
    if not st.session_state.uploaded_file:
        return "Upload papers first before generating hypotheses."

    try:
        with st.spinner("Generating hypotheses..."):
            result = generate_hypotheses_v2(st.session_state.uploaded_file)

        # ensure safe format
        if not result:
            return "No hypotheses generated."

        return result

    except Exception as e:
        return f"Hypothesis generation failed: {e}"

def request_analysis_and_graph() -> None:
    uploaded_files = st.session_state.uploaded_file
    if not uploaded_files:
        st.session_state.feature_outputs["paper_analysis"] = "Upload at least one PDF first."
        st.session_state.feature_outputs["knowledge_graph"] = "Upload at least one PDF first."
        return

    try:
        with st.spinner(f"Analyzing {len(uploaded_files)} file(s) and building knowledge graph..."):
            result = analyze_multiple_papers(uploaded_files)

        st.session_state.summaries_dict = result.get("summaries", {})
        st.session_state.paper_analyses = result.get("paper_analyses", [])
        st.session_state.knowledge_graph = result.get("knowledge_graph", {})

        if st.session_state.paper_analyses:
            st.session_state.feature_outputs["paper_analysis"] = "Structured paper analyses generated successfully."
        else:
            st.session_state.feature_outputs["paper_analysis"] = "Analysis finished, but no structured outputs were returned."

        if st.session_state.knowledge_graph:
            st.session_state.feature_outputs["knowledge_graph"] = "Knowledge graph generated successfully."
        else:
            st.session_state.feature_outputs["knowledge_graph"] = "Graph generation finished, but no graph output was returned."

    except Exception as error:
        error_message = f"Analysis/graph generation failed: {error}"
        st.session_state.paper_analyses = []
        st.session_state.knowledge_graph = {}
        st.session_state.feature_outputs["paper_analysis"] = error_message
        st.session_state.feature_outputs["knowledge_graph"] = error_message


def request_agent_response(user_prompt: str) -> str:
    if not st.session_state.uploaded_file or len(st.session_state.uploaded_file) == 0:
        return "Upload research papers first so the AI agent can ground its answer in the paper content."

    if run_agent is None:
        return missing_backend_message("run_agent", "src/agent.py")

    try:
        return str(run_agent(user_prompt, st.session_state.uploaded_file))
    except TypeError:
        return str(run_agent(user_prompt))
    except Exception as error:
        return f"Agent request failed: {error}"


def request_question_response(user_prompt: str) -> str:
    if qa_answer_question is not None:
        try:
            return str(qa_answer_question(user_prompt, st.session_state.uploaded_file))
        except Exception as error:
            return f"Q&A card request failed: {error}"

    return request_agent_response(user_prompt)


def submit_chat_question(question: str) -> None:
    cleaned_question = question.strip()
    if not cleaned_question:
        return

    st.session_state.chat_messages.append({"role": "user", "content": cleaned_question})
    st.session_state.chat_messages.append(
        {"role": "assistant", "content": request_question_response(cleaned_question)}
    )


def run_feature_prompt(feature: dict[str, str]) -> None:
    key = feature["key"]

    if key == "hypothesis_generation":
        st.session_state.feature_outputs[key] = request_hypotheses()
        return

    prompt = feature.get("agent_prompt", "")
    st.session_state.feature_outputs[key] = request_agent_response(prompt)


def render_academic_summary(paper_title: str, summary_content: str) -> None:
    st.markdown(
        f"""
        <div style="
            background-color: #f8f9fa;
            border-left: 4px solid #0e7c86;
            padding: 2rem;
            border-radius: 8px;
            margin-bottom: 2rem;
        ">
            <div style="
                font-family: 'Fraunces', Georgia, serif;
                font-size: 1.5rem;
                font-weight: 700;
                color: #1a3a42;
                margin-bottom: 1.5rem;
                padding-bottom: 0.75rem;
                border-bottom: 2px solid #0e7c86;
            ">
                {format_html_text(paper_title)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="
            font-family: 'Manrope', 'Trebuchet MS', sans-serif;
            color: #2c3e50;
            margin-left: 2rem;
            margin-right: 2rem;
            line-height: 1.8;
        ">
            {_render_summary_markdown(summary_content)}
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_hypotheses(output):
    st.markdown("### Generated Hypotheses")

    if isinstance(output, list):
        for i, h in enumerate(output, 1):
            st.markdown(f"""
            <div class="analysis-card">
                <div class="panel-eyebrow">Hypothesis {i}</div>
                <h3>{h.get('hypothesis', 'Untitled')}</h3>
                <p><b>Gap:</b> {h.get('based_on_gap', '')}</p>
                <p><b>Reasoning:</b> {h.get('reasoning', '')}</p>
                <p><b>Expected Improvement:</b> {h.get('expected_improvement', '')}</p>
                <p><b>Score:</b> {h.get('score', '')}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.write(output)


def _render_summary_markdown(content: str) -> str:
    escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    formatted = escaped.replace("\n", "<br>")
    return formatted


def render_backend_notice() -> None:
    if summarize_pdf is not None and run_agent is not None:
        return

    st.markdown(
        """
        <div class="backend-note">
            <div class="panel-eyebrow">Backend status</div>
            <h3>UI is ready. Backend hooks can be added next.</h3>
            <p>
            This file assumes that src/summarizer.py exposes summarize_pdf(uploaded_file)
            and src/agent.py exposes run_agent(user_request, uploaded_files=None).
            Until those functions exist, the interface stays usable and will show clear status messages.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_graphviz_from_knowledge_graph(graph: dict) -> str:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    node_styles = {
        "paper": {"shape": "box", "color": "#0e7c86", "style": "filled", "fillcolor": "#dff3f4"},
        "dataset": {"shape": "ellipse", "color": "#3b82f6", "style": "filled", "fillcolor": "#e8f0ff"},
        "model": {"shape": "ellipse", "color": "#8b5cf6", "style": "filled", "fillcolor": "#f0e8ff"},
        "method": {"shape": "ellipse", "color": "#f59e0b", "style": "filled", "fillcolor": "#fff4db"},
        "task": {"shape": "ellipse", "color": "#10b981", "style": "filled", "fillcolor": "#e5fff6"},
        "metric": {"shape": "ellipse", "color": "#ef4444", "style": "filled", "fillcolor": "#ffe8e8"},
    }

    lines = [
        "digraph KnowledgeGraph {",
        'rankdir=LR;',
        'graph [splines=true, overlap=false, fontsize=10];',
        'node [fontname="Helvetica", fontsize=10];',
        'edge [fontname="Helvetica", fontsize=9, color="#6b7280"];',
    ]

    for node in nodes:
        node_id = node.get("id", "").replace("-", "_")
        label = str(node.get("label", "")).replace('"', '\\"')
        node_type = node.get("type", "paper")
        style = node_styles.get(node_type, node_styles["paper"])

        lines.append(
            f'"{node_id}" [label="{label}", shape={style["shape"]}, color="{style["color"]}", style="{style["style"]}", fillcolor="{style["fillcolor"]}"];'
        )

    for edge in edges:
        source = edge.get("source", "").replace("-", "_")
        target = edge.get("target", "").replace("-", "_")
        relation = str(edge.get("relation", "")).replace('"', '\\"')

        if relation == "related_to":
            lines.append(
                f'"{source}" -> "{target}" [label="{relation}", color="#111827", penwidth=2.0, style="dashed"];'
            )
        else:
            lines.append(f'"{source}" -> "{target}" [label="{relation}"];')

    lines.append("}")
    return "\n".join(lines)


def render_tag_list(items, empty_text="No items available."):
    if not items:
        st.caption(empty_text)
        return
    html_tags = "".join(
        [f'<span class="mini-tag">{format_html_text(item)}</span>' for item in items if item]
    )
    st.markdown(html_tags, unsafe_allow_html=True)


def render_analysis_card(analysis: dict) -> None:
    title = analysis.get("title") or analysis.get("file_name") or "Untitled Paper"
    paper_type = analysis.get("paper_type", "unclear")
    proposed_method = analysis.get("proposed_method", "")
    problem_statement = analysis.get("problem_statement", "")

    st.markdown(
        f"""
        <div class="analysis-card">
            <div class="panel-eyebrow">Paper analysis</div>
            <h3>{format_html_text(title)}</h3>
            <p class="subtle-text"><strong>Type:</strong> {format_html_text(paper_type)}</p>
            <p class="subtle-text"><strong>Problem:</strong> {format_html_text(problem_statement or "Not clearly extracted.")}</p>
            <p class="subtle-text"><strong>Method:</strong> {format_html_text(proposed_method or "Not clearly extracted.")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### Datasets")
        render_tag_list(analysis.get("datasets", []), "No datasets extracted.")

        st.markdown("#### Architecture Changes")
        for item in analysis.get("architecture_changes", []):
            st.markdown(f"- {item}")
        if not analysis.get("architecture_changes", []):
            st.caption("No clear architecture changes extracted.")

        st.markdown("#### Training Setup")
        for item in analysis.get("training_setup", []):
            st.markdown(f"- {item}")
        if not analysis.get("training_setup", []):
            st.caption("No training setup extracted.")

        st.markdown("#### Strengths")
        for item in analysis.get("strengths", []):
            st.markdown(f"- {item}")
        if not analysis.get("strengths", []):
            st.caption("No strengths extracted.")

    with col2:
        st.markdown("#### Results")
        for item in analysis.get("results", []):
            st.markdown(f"- {item}")
        if not analysis.get("results", []):
            st.caption("No result statements extracted.")

        st.markdown("#### Claimed Novelty")
        for item in analysis.get("claimed_novelty", []):
            st.markdown(f"- {item}")
        if not analysis.get("claimed_novelty", []):
            st.caption("No novelty extracted.")

        st.markdown("#### Limitations")
        for item in analysis.get("limitations", []):
            st.markdown(f"- {item}")
        if not analysis.get("limitations", []):
            st.caption("No limitations extracted.")

        st.markdown("#### Future Work")
        for item in analysis.get("future_work", []):
            st.markdown(f"- {item}")
        if not analysis.get("future_work", []):
            st.caption("No future work extracted.")

    st.markdown("#### Structured Result Records")
    result_records = analysis.get("result_records", [])
    if result_records:
        st.table(result_records)
    else:
        st.caption("No structured result records extracted.")

    with st.expander("Show full raw analysis JSON"):
        st.json(analysis)


def render_shared_elements(shared_elements: dict) -> None:
    st.markdown("### Shared Elements Across Papers")
    if not shared_elements:
        st.caption("No shared elements were returned.")
        return

    for category, values in shared_elements.items():
        st.markdown(f"#### {category.replace('_', ' ').title()}")
        if not values:
            st.caption("No shared items.")
            continue

        for item, papers in values.items():
            papers_text = ", ".join(papers)
            st.markdown(f"- **{item}** → {papers_text}")


def render_per_paper_comparison(entries: list) -> None:
    st.markdown("### Per-Paper Unique Comparison")
    if not entries:
        st.caption("No per-paper comparison entries were returned.")
        return

    for entry in entries:
        st.markdown(
            f"""
            <div class="comparison-card">
                <div class="panel-eyebrow">Unique paper profile</div>
                <h3>{format_html_text(entry.get("title") or entry.get("file_name") or "Untitled Paper")}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("#### Unique Methods")
            render_tag_list(entry.get("unique_methods", []), "No unique methods identified.")

            st.markdown("#### Unique Datasets")
            render_tag_list(entry.get("unique_datasets", []), "No unique datasets identified.")

        with col2:
            st.markdown("#### Unique Tasks")
            render_tag_list(entry.get("unique_tasks", []), "No unique tasks identified.")

            st.markdown("#### Standout Points")
            standout_points = entry.get("standout_points", [])
            if standout_points:
                for item in standout_points:
                    st.markdown(f"- {item}")
            else:
                st.caption("No standout points extracted.")


def render_comparative_insights(insights: list) -> None:
    st.markdown("### Comparative Insights")
    if not insights:
        st.caption("No comparative insights were returned.")
        return

    for insight in insights:
        category = insight.get("category", "general")
        statement = insight.get("statement", "")
        supporting_papers = insight.get("supporting_papers", [])

        st.markdown(
            f"""
            <div class="comparison-card">
                <div class="panel-eyebrow">{format_html_text(category)}</div>
                <h3>{format_html_text(statement or "No statement available.")}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if supporting_papers:
            st.markdown("**Supporting Papers:**")
            render_tag_list(supporting_papers, "No supporting papers listed.")


def render_best_paper_candidates(candidates: list) -> None:
    st.markdown("### Best Paper Candidates by Criterion")
    if not candidates:
        st.caption("No best-paper candidates were returned.")
        return

    for candidate in candidates:
        criterion = candidate.get("criterion", "unspecified criterion")
        title = candidate.get("title") or candidate.get("file_name") or "Untitled Paper"
        justification = candidate.get("justification", "")

        st.markdown(
            f"""
            <div class="comparison-card">
                <div class="panel-eyebrow">{format_html_text(criterion)}</div>
                <h3>{format_html_text(title)}</h3>
                <p class="subtle-text">{format_html_text(justification or "No justification available.")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_hero_section() -> None:
    left_column, right_column = st.columns([1.2, 0.9], gap="large")

    with left_column:
        st.markdown(
            """
            <div class="hero-copy">
                <div class="section-kicker">Academic AI Workspace</div>
                <h1>Intelligent Research Co-Pilot</h1>
                <p>
                    An AI assistant for analyzing research papers. Upload one or more papers, generate summaries,
                    ask grounded questions, build structured per-paper analyses, and discover cross-paper links.
                </p>
                <div class="hero-stats">
                    <div class="hero-stat">
                        <strong>Modular architecture</strong>
                        <span>UI stays inside app.py while RAG and agent logic live under the src folder.</span>
                    </div>
                    <div class="hero-stat">
                        <strong>Multi-paper workflows</strong>
                        <span>One interface for upload, summarization, paper Q&amp;A, structured analysis, and knowledge graphs.</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_column:
        st.markdown(
            """
            <div class="assistant-visual">
                <div class="assistant-status">
                    <span class="assistant-status-dot"></span>
                    <span>Chat assistant ready for paper analysis</span>
                </div>
                <div class="assistant-window">
                    <div class="visual-bubble user">Compare the methods used across these papers.</div>
                    <div class="visual-bubble assistant">I can summarize, answer questions, generate structured paper analyses, and build a knowledge graph.</div>
                    <div class="visual-bubble user">What is the strongest common research direction?</div>
                </div>
                <div class="assistant-orb">AI</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_upload_section() -> None:
    render_section_header(
        "Step 1",
        "Step 1: Upload Your Research Papers",
        "Upload your PDF files first. Then use the feature cards below to navigate to summarization, Q&A, structured analysis, or knowledge graph workflows.",
    )

    left_column, right_column = st.columns([1.03, 0.97], gap="large")

    with left_column:
        st.markdown(
            """
            <div class="workspace-card">
                <div class="panel-eyebrow">Document input</div>
                <h3>Load your papers (PDF)</h3>
                <p>
                    This upload section is always visible so you can switch features while keeping
                    the same paper context.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Upload PDF papers (1-5 files)",
            type=["pdf"],
            key="paper_uploader",
            help="Choose one or more research papers in PDF format (up to 5 files).",
            accept_multiple_files=True,
        )
        sync_uploaded_file(uploaded_files)

        if st.session_state.uploaded_file and len(st.session_state.uploaded_file) > 0:
            for file_name in st.session_state.uploaded_file_name:
                st.markdown(
                    f"<div class=\"file-pill\">Uploaded: {format_html_text(file_name)}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
                <div class="placeholder-box">
                    <div class="panel-eyebrow">Status</div>
                    <h3>No files uploaded yet</h3>
                    <p>Upload 1-5 PDFs to unlock summary generation, structured paper analysis, knowledge graph generation, and paper-grounded question answering.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right_column:
        file_count = len(st.session_state.uploaded_file) if st.session_state.uploaded_file else 0
        readiness_text = (
            f"{file_count} file(s) uploaded and ready. Select a feature card below to start analysis."
            if file_count > 0
            else "After uploading papers, move to Step 2 and choose a feature card to open its workspace."
        )
        st.markdown(
            f"""
            <div class="summary-box">
                <div class="summary-label">Document status</div>
                <h3>Paper readiness</h3>
                <p>{format_html_text(readiness_text)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_feature_card(feature: dict[str, str]) -> None:
    st.markdown(
        f"""
        <div class="feature-card">
            <div class="feature-eyebrow">{feature['eyebrow']}</div>
            <h3>{feature['title']}</h3>
            <p>{feature['description']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.button(
        feature["button"],
        key=f"feature_{feature['key']}",
        use_container_width=True,
        on_click=set_active_feature,
        args=(feature["key"],),
    )


def render_features_section() -> None:
    render_section_header(
        "Step 2",
        "Step 2: Choose a Feature",
        "All feature cards stay visible for navigation. Click any card to open its dedicated workspace below.",
    )

    for row_start in range(0, len(FEATURES), 3):
        columns = st.columns(3, gap="large")
        for column, feature in zip(columns, FEATURES[row_start : row_start + 3]):
            with column:
                render_feature_card(feature)


def render_summary_workspace() -> None:
    render_section_header(
        "Step 3",
        "Selected Workspace: Summarize Paper",
        "Generate a summary for the currently uploaded document(s). The output appears in the result panel below.",
    )

    left_column, right_column = st.columns([1.03, 0.97], gap="large")

    with left_column:
        st.markdown(
            """
            <div class="workspace-card">
                <div class="panel-eyebrow">Summary action</div>
                <h3>Generate paper summary</h3>
                <p>
                    Click the button to call summarize_pdf(uploaded_file) or summarize_multiple_pdfs(uploaded_files)
                    from your backend module.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.uploaded_file is not None:
            active_label = (
                ", ".join(st.session_state.uploaded_file_name)
                if isinstance(st.session_state.uploaded_file_name, list)
                else st.session_state.uploaded_file_name
            )
            st.markdown(
                f"<div class=\"file-pill\">Active file(s): {format_html_text(active_label)}</div>",
                unsafe_allow_html=True,
            )

        file_count = len(st.session_state.uploaded_file) if st.session_state.uploaded_file else 0
        if st.button(
            "Generate Summaries",
            key="generate_summary",
            use_container_width=True,
            disabled=file_count == 0,
        ):
            request_summary()
            st.rerun()

    with right_column:
        if st.session_state.summaries_dict:
            for file_name, summary in st.session_state.summaries_dict.items():
                render_academic_summary(file_name, summary)
        else:
            summary_text = st.session_state.summary_text or (
                "Your generated summaries will appear here after you upload PDFs and click Generate Summaries."
            )
            st.markdown(
                f"""
                <div class="summary-box">
                    <div class="summary-label">Summary output</div>
                    <h3>{format_html_text(get_active_paper_title())}</h3>
                    <p>{format_html_text(summary_text)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_paper_analysis_workspace() -> None:
    render_section_header(
        "Step 3",
        "Selected Workspace: Paper Analysis Agent",
        "Generate structured analysis for each uploaded paper.",
    )

    if not st.session_state.uploaded_file or len(st.session_state.uploaded_file) == 0:
        st.markdown(
            """
            <div class="placeholder-box">
                <div class="panel-eyebrow">Upload needed</div>
                <h3>Upload papers first</h3>
                <p>Go to Step 1, upload 1-5 PDFs, then come back here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if st.button("Generate Paper Analyses", key="generate_paper_analysis", use_container_width=True):
        request_analysis_and_graph()
        st.rerun()

    status_message = st.session_state.feature_outputs.get("paper_analysis", "")
    if status_message:
        st.info(status_message)

    analyses = st.session_state.paper_analyses
    if not analyses:
        st.markdown(
            """
            <div class="placeholder-box">
                <div class="panel-eyebrow">No analysis yet</div>
                <h3>Structured outputs will appear here</h3>
                <p>Click the button above to generate per-paper analysis.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for analysis in analyses:
        render_analysis_card(analysis)


def render_knowledge_graph_workspace() -> None:
    render_section_header(
        "Step 3",
        "Selected Workspace: Knowledge Graph Agent",
        "Build a cross-paper graph showing links between papers, datasets, models, methods, tasks, and metrics.",
    )

    if not st.session_state.uploaded_file or len(st.session_state.uploaded_file) == 0:
        st.markdown(
            """
            <div class="placeholder-box">
                <div class="panel-eyebrow">Upload needed</div>
                <h3>Upload papers first</h3>
                <p>Go to Step 1, upload 1-5 PDFs, then come back here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if st.button("Generate Knowledge Graph", key="generate_knowledge_graph", use_container_width=True):
        request_analysis_and_graph()
        st.rerun()

    status_message = st.session_state.feature_outputs.get("knowledge_graph", "")
    if status_message:
        st.info(status_message)

    graph = st.session_state.knowledge_graph
    if not graph:
        st.markdown(
            """
            <div class="placeholder-box">
                <div class="panel-eyebrow">No graph yet</div>
                <h3>Knowledge graph output will appear here</h3>
                <p>Click the button above to generate graph nodes, edges, and insights.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.subheader("Knowledge Graph Visualization")
    dot_graph = build_graphviz_from_knowledge_graph(graph)
    st.graphviz_chart(dot_graph, use_container_width=True)

    st.markdown("---")
    render_shared_elements(graph.get("shared_elements", {}))

    st.markdown("---")
    render_per_paper_comparison(graph.get("per_paper_comparison", []))

    st.markdown("---")
    render_comparative_insights(graph.get("comparative_insights", []))

    st.markdown("---")
    render_best_paper_candidates(graph.get("best_paper_candidates", []))

    simple_insights = graph.get("insights", [])
    if simple_insights:
        st.markdown("### General Insights")
        for insight in simple_insights:
            st.markdown(f"- {insight}")

    with st.expander("Show Graph JSON"):
        st.json(graph)


def render_question_workspace() -> None:
    render_section_header(
        "Step 3",
        "Selected Workspace: Ask Questions from Paper",
        "Ask paper-grounded questions below. Conversation and answers are kept in session history.",
    )

    if not st.session_state.uploaded_file or len(st.session_state.uploaded_file) == 0:
        st.markdown(
            """
            <div class="placeholder-box">
                <div class="panel-eyebrow">Upload needed</div>
                <h3>Upload papers first</h3>
                <p>Go to Step 1, upload 1-5 PDFs, then come back to ask questions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        """
        <div class="chat-shell">
            <div class="panel-eyebrow">Question answering</div>
            <h3>Paper Q&amp;A</h3>
            <p>Type a question and get an answer from the backend agent using the uploaded paper context.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for message in st.session_state.chat_messages:
        role_label = "Assistant" if message["role"] == "assistant" else "You"
        st.markdown(
            f"""
            <div class="message-card {message['role']}">
                <div class="message-role">{role_label}</div>
                <div class="message-text">{format_html_text(message['content'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.form("workspace_question_form", clear_on_submit=True):
        question = st.text_input(
            "Ask a question about the paper",
            placeholder="What are the paper's key findings?",
        )
        send = st.form_submit_button("Ask AI Agent", use_container_width=True)

    if send and question.strip():
        submit_chat_question(question)
        st.session_state.feature_outputs["ask_questions"] = st.session_state.chat_messages[-1]["content"]
        st.rerun()


def render_agent_workspace(feature: dict[str, str]) -> None:
    render_section_header(
        "Step 3",
        f"Selected Workspace: {feature['title']}",
        "This section forwards a task-specific prompt to the backend agent, which can decide which tool should handle the request.",
    )

    st.markdown(
        f"""
        <div class="workspace-card">
            <div class="panel-eyebrow">Agent task</div>
            <h3>{feature['title']}</h3>
            <p>{feature['description']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Run with AI Agent", key=f"run_{feature['key']}", use_container_width=True):
        run_feature_prompt(feature)

    output = st.session_state.feature_outputs.get(
        feature["key"],
        "The backend agent response will appear here after you run this workflow.",
    )
    st.markdown(
        f"""
        <div class="placeholder-box">
            <div class="panel-eyebrow">Current output</div>
            <h3>{feature['title']}</h3>
            <p>{format_html_text(output)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workspace_section() -> None:
    active_feature = st.session_state.active_feature

    if active_feature is None:
        render_section_header(
            "Step 3",
            "Selected Feature Workspace",
            "Choose a feature card in Step 2. The selected tool and its output will appear here.",
        )
        st.markdown(
            """
            <div class="placeholder-box">
                <div class="panel-eyebrow">Awaiting selection</div>
                <h3>No feature selected</h3>
                <p>Select one of the cards above to open its workspace below.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if active_feature == "summarize_paper":
        render_summary_workspace()
        return

    if active_feature == "paper_analysis":
        render_paper_analysis_workspace()
        return

    if active_feature == "knowledge_graph":
        render_knowledge_graph_workspace()
        return

    if active_feature == "ask_questions":
        render_question_workspace()
        return

    if active_feature == "hypothesis_generation":
        render_section_header("Step 3", "Hypothesis Generation", "AI-generated research hypotheses")

        if st.button("Generate Hypotheses"):
            st.session_state.feature_outputs["hypothesis_generation"] = request_hypotheses()
            st.rerun()

        output = st.session_state.feature_outputs.get("hypothesis_generation")

        if isinstance(output, str):
            st.error(output)
        elif output:
            render_hypotheses(output)
        else:
            st.info("No hypotheses generated yet.")
        return

    for feature in FEATURES:
        if feature["key"] == active_feature:
            render_agent_workspace(feature)
            return


def main() -> None:
    configure_page()
    initialize_session_state()
    render_styles()
    render_hero_section()
    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)
    render_upload_section()
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    render_features_section()
    st.markdown("<div style='height: 1.6rem;'></div>", unsafe_allow_html=True)
    render_workspace_section()


if __name__ == "__main__":
    main()