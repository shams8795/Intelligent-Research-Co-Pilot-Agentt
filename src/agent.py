"""LangGraph ReAct agent for research paper workflows.

Uses create_react_agent from langgraph.prebuilt so the LLM decides
which tool to call based on the user request.
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from src.tools import get_agent_tools


load_dotenv()

_SYSTEM_PROMPT = (
    "You are an Intelligent Research Co-Pilot specialised in analysing academic research papers.\n"
    "You may have access to one or more uploaded research papers.\n"
    "Use the available tools whenever the user asks for summaries, structured paper analysis, "
    "cross-paper comparisons, or knowledge graph connections.\n"
    "Always ground your answers in the uploaded paper content.\n"
    "Be technical, precise, and helpful."
)


def _get_llm() -> ChatGroq:
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(model=model, temperature=0)


def run_agent(user_request: str, uploaded_files=None) -> str:
    """Run the LangGraph ReAct agent on a user request."""
    if not isinstance(user_request, str) or not user_request.strip():
        return "Please provide a valid request."

    if not os.getenv("GROQ_API_KEY"):
        return "GROQ_API_KEY is missing. Add it to your .env file and restart."

    files = uploaded_files or []
    tools = get_agent_tools(files)
    llm = _get_llm()

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=_SYSTEM_PROMPT,
    )

    try:
        result = agent.invoke({"messages": [HumanMessage(content=user_request)]})
    except Exception as error:
        return f"Agent execution failed: {error}"

    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        content = getattr(last, "content", None)
        if content:
            return str(content).strip()

    return "The agent did not return a response. Please try again."