"""
Financial-performance agent with Streamlit UI
Updated: optional CSV upload + PDF download
"""

# ─── Standard library ───────────────────────────────────────────────────────────
from __future__ import annotations
import os, json, re
from io import StringIO, BytesIO
from typing import List, TypedDict

# ─── Third-party ────────────────────────────────────────────────────────────────
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from tavily import TavilyClient
import streamlit as st
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# ─── API keys ───────────────────────────────────────────────────────────────────
load_dotenv()
def _clean(v: str | None) -> str: return (v or "").strip().strip('"\'“”')
OPENAI_API_KEY = _clean(os.getenv("OPENAI_API_KEY"))
TAVILY_API_KEY = _clean(os.getenv("TAVILY_API_KEY"))

# ─── External clients ───────────────────────────────────────────────────────────
model = ChatOpenAI(model_name="gpt-4o-mini",
                   openai_api_key=OPENAI_API_KEY,
                   temperature=0.2)
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# ─── LangGraph state schema ─────────────────────────────────────────────────────
class AgentState(TypedDict, total=False):
    task: str
    competitors: List[str]
    csv_file: str               # raw CSV text
    financial_data: str
    analysis: str
    content: List[str]
    comparison: str
    feedback: str
    report: str
    revision_number: int
    max_revisions: int

# ─── Helper pydantic model ──────────────────────────────────────────────────────
class Queries(BaseModel): queries: List[str]

# ─── Prompts ────────────────────────────────────────────────────────────────────
GATHER_FINANCIALS_PROMPT   = "Summarise key metrics in the CSV below."
ANALYZE_DATA_PROMPT        = "Deep-dive analysis of the company’s performance:"
RESEARCH_COMP_PROMPT       = "Generate ≤3 concise search queries for benchmarking."
COMPARE_PROMPT             = "Compare the target company with each competitor:"
FEEDBACK_PROMPT            = "Provide detailed reviewer feedback on the draft."
WRITE_REPORT_PROMPT        = "Write an investor-ready final report."
CRITIQUE_SEARCH_PROMPT     = "Generate ≤3 queries to address reviewer comments."

# ─── LangGraph nodes ────────────────────────────────────────────────────────────
def gather_financials_node(state: AgentState) -> AgentState:
    csv_txt = state.get("csv_file", "")
    if not csv_txt:                                        # no upload
        return {}                                          # skip analysis path
    df = pd.read_csv(StringIO(csv_txt))
    msgs = [SystemMessage(GATHER_FINANCIALS_PROMPT),
            HumanMessage(df.to_markdown(index=False))]
    resp = model.invoke(msgs).content
    return {"financial_data": resp}

def analyze_data_node(state: AgentState) -> AgentState:
    if not state.get("financial_data"):                    # nothing to analyse
        return {"analysis": ""}
    msgs = [SystemMessage(ANALYZE_DATA_PROMPT),
            HumanMessage(state["financial_data"])]
    resp = model.invoke(msgs).content
    return {"analysis": resp}

def research_competitors_node(state: AgentState) -> AgentState:
    docs: list[str] = state.get("content", [])
    for name in state["competitors"]:
        q_obj: Queries = model.with_structured_output(Queries).invoke(
            [SystemMessage(RESEARCH_COMP_PROMPT), HumanMessage(name)])
        for q in q_obj.queries:
            for r in tavily.search(q, max_results=3)["results"]:
                docs.append(r["content"])
    return {"content": docs}

def compare_performance_node(state: AgentState) -> AgentState:
    body = "\n\n".join(state.get("content", []))
    msgs = [SystemMessage(f"{COMPARE_PROMPT}\n\n{body}"),
            HumanMessage(state.get("analysis", ""))]
    draft = model.invoke(msgs).content
    return {"comparison": draft,
            "revision_number": state.get("revision_number", 1) + 1}

def collect_feedback_node(state: AgentState) -> AgentState:
    msgs = [SystemMessage(FEEDBACK_PROMPT),
            HumanMessage(state["comparison"])]
    return {"feedback": model.invoke(msgs).content}

def research_critique_node(state: AgentState) -> AgentState:
    q_obj: Queries = model.with_structured_output(Queries).invoke(
        [SystemMessage(CRITIQUE_SEARCH_PROMPT),
         HumanMessage(state["feedback"])])
    docs = state.get("content", [])
    for q in q_obj.queries:
        docs.extend(r["content"] for r in tavily.search(q, max_results=3)["results"])
    return {"content": docs}

def write_report_node(state: AgentState) -> AgentState:
    msgs = [SystemMessage(WRITE_REPORT_PROMPT),
            HumanMessage(state["comparison"])]
    return {"report": model.invoke(msgs).content}

# ─── Flow control ───────────────────────────────────────────────────────────────
def route_after_compare(state: AgentState) -> str:
    return "write_report" if state["revision_number"] > state["max_revisions"] else "collect_feedback"

# ─── Build the graph ────────────────────────────────────────────────────────────
memory = SqliteSaver.from_conn_string(":memory:")
builder = StateGraph(AgentState)

builder.add_node("gather_financials", gather_financials_node)
builder.add_node("analyze_data",      analyze_data_node)
builder.add_node("research_comp",     research_competitors_node)
builder.add_node("compare",           compare_performance_node)

# 🔧 rename the node from "feedback" → "collect_feedback"
builder.add_node("collect_feedback",  collect_feedback_node)
builder.add_node("critique_search",   research_critique_node)
builder.add_node("write_report",      write_report_node)

builder.set_entry_point("gather_financials")

builder.add_edge("gather_financials", "analyze_data")
builder.add_edge("analyze_data",      "research_comp")
builder.add_edge("research_comp",     "compare")
builder.add_edge("collect_feedback",  "critique_search")   

# conditional routing
builder.add_conditional_edges(
    "compare",
    route_after_compare,
    {
        "collect_feedback": "collect_feedback",            
        "write_report": "write_report",
    },
)

graph = builder.compile(checkpointer=memory)


# ─── PDF helper ────────────────────────────────────────────────────────────────
def make_pdf(text: str) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER,
                            leftMargin=50, rightMargin=50,
                            topMargin=50, bottomMargin=50)
    style = getSampleStyleSheet()["Normal"]
    story = []
    for para in text.split("\n\n"):
        story.extend([Paragraph(para.strip(), style), Spacer(1, 12)])
    doc.build(story)
    buf.seek(0)
    return buf.read()

# ─── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Financial-Performance Agent", layout="wide")

def main() -> None:
    st.title("📊 Financial Performance Reporting Agent")

    task = st.text_input("Task",
        "Compare the financial performance of the listed companies")
    companies = st.text_area("Company names (one per line)").splitlines()
    max_rev = st.number_input("Maximum revision cycles", 1, 10, value=2)
    csv_file = st.file_uploader("Optional: Upload CSV for the **first** company", type="csv")

    if st.button("Run analysis"):
        csv_txt = csv_file.getvalue().decode() if csv_file else ""
        init: AgentState = {
            "task": task,
            "competitors": [c.strip() for c in companies if c.strip()],
            "csv_file": csv_txt,
            "content": [],
            "revision_number": 1,
            "max_revisions": int(max_rev),
        }

        with SqliteSaver.from_conn_string(":memory:") as cp:
            final = None
            for s in builder.compile(cp).stream(init, {"configurable": {"thread_id": "ui"}}):
                st.write(s)
                final = s

        if final and final.get("report"):
            st.subheader("💼 Final Report")
            rpt = final["report"]
            st.write(rpt)

            pdf_bytes = make_pdf(rpt)
            st.download_button("⬇️ Download PDF", pdf_bytes,
                               file_name="financial_report.pdf",
                               mime="application/pdf")

if __name__ == "__main__":
    main()
