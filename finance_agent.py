"""
Financial-performance agent with Streamlit UI
Compatible with:
  • Python ≥ 3.9 (tested on 3.12)
  • langchain-openai ≥ 0.1.0
  • langgraph ≥ 0.0.41
  • streamlit ≥ 1.33
  • tavily-python ≥ 0.3.2
"""

# ─── Standard library ───────────────────────────────────────────────────────────
from __future__ import annotations

import os
import re
import json
from io import StringIO
from typing import List, TypedDict

# ─── Third-party ────────────────────────────────────────────────────────────────
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from tavily import TavilyClient
import streamlit as st


# ─── Environment & keys ─────────────────────────────────────────────────────────
load_dotenv()

def _clean(value: str | None) -> str:
    return (value or "").strip().strip('"\''"“”")

OPENAI_API_KEY = _clean(os.getenv("OPENAI_API_KEY"))
TAVILY_API_KEY = _clean(os.getenv("TAVILY_API_KEY"))

# ─── External clients ───────────────────────────────────────────────────────────
model = ChatOpenAI(
    model_name="gpt-4o-mini",        # >>> changed: newer model; adjust if desired
    openai_api_key=OPENAI_API_KEY,
    temperature=0.2,
)
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# ─── LangGraph state schema ─────────────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    task: str
    competitors: List[str]
    csv_file: str
    financial_data: str
    analysis: str
    competitor_data: str          # consolidated competitor docs
    comparison: str
    feedback: str
    content: List[str]            # scratch pad for retrieved docs
    report: str
    revision_number: int
    max_revisions: int

# ─── Helper model for structured LLM output ─────────────────────────────────────
class Queries(BaseModel):
    queries: List[str]

# ─── Prompts ────────────────────────────────────────────────────────────────────
GATHER_FINANCIALS_PROMPT = (
    "You are an expert financial analyst. "
    "Summarise the key metrics, trends, and red-flags in the following CSV data."
)

ANALYZE_DATA_PROMPT = (
    "You are an expert financial analyst. Provide a deep-dive analysis of the "
    "company's performance based on the extracted metrics. Be concise but thorough."
)

RESEARCH_COMPETITORS_PROMPT = (
    "Generate up to three **concise** web-search queries I can use to gather "
    "information on similar companies for benchmarking."
)

COMPARE_PERFORMANCE_PROMPT = (
    "Compare the target company with each competitor below. "
    "Highlight relative strengths, weaknesses, and notable ratios."
)

FEEDBACK_PROMPT = (
    "You are a picky reviewer. Provide constructive, detailed feedback on the "
    "draft comparison report."
)

WRITE_REPORT_PROMPT = (
    "Write a polished, investor-ready report incorporating the analysis, "
    "benchmarking, and reviewer feedback."
)

RESEARCH_CRITIQUE_PROMPT = (
    "Generate up to three targeted search queries that would help address the "
    "specific reviewer comments."
)

# ─── LangGraph nodes ────────────────────────────────────────────────────────────
def gather_financials_node(state: AgentState) -> AgentState:
    df = pd.read_csv(StringIO(state["csv_file"]))
    messages = [
        SystemMessage(content=GATHER_FINANCIALS_PROMPT),
        HumanMessage(content=df.to_markdown(index=False)),
    ]
    response = model.invoke(messages)
    return {"financial_data": response.content}

def analyze_data_node(state: AgentState) -> AgentState:
    messages = [
        SystemMessage(content=ANALYZE_DATA_PROMPT),
        HumanMessage(content=state["financial_data"]),
    ]
    response = model.invoke(messages)
    return {"analysis": response.content}

def research_competitors_node(state: AgentState) -> AgentState:
    docs: list[str] = state.get("content", [])
    for name in state["competitors"]:
        chain = model.with_structured_output(Queries)
        qry_obj: Queries = chain.invoke(
            [SystemMessage(content=RESEARCH_COMPETITORS_PROMPT),
             HumanMessage(content=name)]
        )
        for q in qry_obj.queries:
            search_resp = tavily.search(q, max_results=3)
            docs.extend(r["content"] for r in search_resp["results"])
    return {"content": docs}

def compare_performance_node(state: AgentState) -> AgentState:
    body = "\n\n".join(state.get("content", []))
    messages = [
        SystemMessage(content=f"{COMPARE_PERFORMANCE_PROMPT}\n\n{body}"),
        HumanMessage(content=state["analysis"]),
    ]
    draft = model.invoke(messages).content
    next_rev = state.get("revision_number", 1) + 1
    return {"comparison": draft, "revision_number": next_rev}

def collect_feedback_node(state: AgentState) -> AgentState:
    messages = [
        SystemMessage(content=FEEDBACK_PROMPT),
        HumanMessage(content=state["comparison"]),
    ]
    fb = model.invoke(messages).content
    return {"feedback": fb}

def research_critique_node(state: AgentState) -> AgentState:
    chain = model.with_structured_output(Queries)
    qry_obj: Queries = chain.invoke(
        [SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
         HumanMessage(content=state["feedback"])]
    )
    docs = state.get("content", [])
    for q in qry_obj.queries:
        search_resp = tavily.search(q, max_results=3)
        docs.extend(r["content"] for r in search_resp["results"])
    return {"content": docs}

def write_report_node(state: AgentState) -> AgentState:
    messages = [
        SystemMessage(content=WRITE_REPORT_PROMPT),
        HumanMessage(content=state["comparison"]),
    ]
    final = model.invoke(messages).content
    return {"report": final}

# ─── Flow control ───────────────────────────────────────────────────────────────
def route_after_compare(state: AgentState) -> str:
    """Decide whether to loop again or finish."""
    if state["revision_number"] > state["max_revisions"]:
        return "write_report"
    return "collect_feedback"

memory = SqliteSaver.from_conn_string(":memory:")
builder = StateGraph(AgentState)

builder.add_node("gather_financials",     gather_financials_node)
builder.add_node("analyze_data",          analyze_data_node)
builder.add_node("research_competitors",  research_competitors_node)
builder.add_node("compare_performance",   compare_performance_node)
builder.add_node("collect_feedback",      collect_feedback_node)
builder.add_node("research_critique",     research_critique_node)
builder.add_node("write_report",          write_report_node)

builder.set_entry_point("gather_financials")

builder.add_edge("gather_financials",    "analyze_data")
builder.add_edge("analyze_data",         "research_competitors")
builder.add_edge("research_competitors", "compare_performance")
builder.add_edge("collect_feedback",     "research_critique")
builder.add_edge("research_critique",    "compare_performance")

# >>> changed: conditional routing
builder.add_conditional_edges(
    "compare_performance",
    route_after_compare,
    {
        "collect_feedback": "collect_feedback",
        "write_report": "write_report",
    },
)

graph = builder.compile(checkpointer=memory)

# ─── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Financial-Performance Agent", layout="wide")

def main() -> None:
    st.title("📊 Financial Performance Reporting Agent")

    task       = st.text_input("Task",
        "Analyse our company's financial performance versus competitors")
    competitors = st.text_area(
        "Competitor names (one per line)").splitlines()
    max_revs   = st.number_input("Maximum revision cycles", 1, 10, value=2)
    csv_file   = st.file_uploader(
        "Upload the company's CSV financial data", type="csv")

    if st.button("Run analysis") and csv_file:
        csv_data = csv_file.getvalue().decode("utf-8")

        init_state: AgentState = {
            "task": task,
            "competitors": [c.strip() for c in competitors if c.strip()],
            "csv_file": csv_data,
            "content": [],
            "revision_number": 1,
            "max_revisions": int(max_revs),
        }

        with SqliteSaver.from_conn_string(":memory:") as cp:
            exec_graph = builder.compile(checkpointer=cp)
            final: AgentState | None = None

            for state in exec_graph.stream(init_state, {"configurable": {"thread_id": "ui"}}):
                st.write(state)
                final = state

        if final and final.get("report"):
            st.subheader("💼 Final Report")
            st.write(final["report"])

if __name__ == "__main__":
    main()
