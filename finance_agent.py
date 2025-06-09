import os
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv
from openai import OpenAI
import re
import operator
import json
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
import pandas as pd
from io import StringIO
from tavily import TavilyClient
from typing import TypedDict
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage

memory = SqliteSaver.from_conn_string(":memory:")

load_dotenv()

open_key = os.getenv("OPENAI_API_KEY")
openai_key = open_key.strip().strip('"').strip("'").replace("“", "").replace("”", "")

tavily_key = os.getenv("TAVILY_API_KEY")

llm_name="gpt-3.5-turbo"

model = ChatOpenAI(api_key=openai_key, model=llm_name)
tavily = TavilyClient(api_key=tavily_key)

class AgentState(TypedDict):
    task : str
    competitors : List[str]
    csv_file : str
    financial_data :str
    analysis: str
    competitor_data : str
    comparison : str
    feedback : str
    content: List[str]
    report : str
    revision_number: int
    max_revision: str
    
class Queries(BaseModel):
    queries: List[str]
    
# Define the prompts for each node - IMPROVE AS NEEDED
GATHER_FINANCIALS_PROMPT = """You are an expert financial analyst. Gather the financial data for the given company. Provide detailed financial data."""
ANALYZE_DATA_PROMPT = """You are an expert financial analyst. Analyze the provided financial data and provide detailed insights and analysis."""
RESEARCH_COMPETITORS_PROMPT = """You are a researcher tasked with providing information about similar companies for performance comparison. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""
COMPETE_PERFORMANCE_PROMPT = """You are an expert financial analyst. Compare the financial performance of the given company with its competitors based on the provided data.
**MAKE SURE TO INCLUDE THE NAMES OF THE COMPETITORS IN THE COMPARISON.**"""
FEEDBACK_PROMPT = """You are a reviewer. Provide detailed feedback and critique for the provided financial comparison report. Include any additional information or revisions needed."""
WRITE_REPORT_PROMPT = """You are a financial report writer. Write a comprehensive financial report based on the analysis, competitor research, comparison, and feedback provided."""
RESEARCH_CRITIQUE_PROMPT = """You are a researcher tasked with providing information to address the provided critique. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""


def gather_financials_node(state:AgentState):
    csv_file = state['csv_file']
    df = pd.read_csv(StringIO(csv_file))
    
    financial_data_str = df.to_string(index=False)
    
    combine_content = (
        f"{state['task']}\n\nHere is the financial data:\n\n{financial_data_str}"
    )
    
    messages = [
        SystemMessage(content = GATHER_FINANCIALS_PROMPT),
        HumanMessage(content = combine_content),
    ]
    
    response = model.invoke(messages)
    return {"financial_data": response.content }
    
def analyze_data_node(state:AgentState):
    messages = [
        SystemMessage(content=ANALYZE_DATA_PROMPT),
        HumanMessage(content=state['financial_data'])
    ]
    response = model.invoke(messages)
    return {"Analysis": response.content}

def research_competitors_node(state:AgentState):
    content = state.get("content", [])
    for competitor in state["competitors"]:
        queries = model.with_structured_output(Queries).invoke(
            SystemMessage(content=RESEARCH_COMPETITORS_PROMPT),
            HumanMessage(content=competitor),
        )
        for q in queries.queries:
            response = tavily.search(query=q, max_results=2)
            for r in response["results"]:
                content.append(r["content"])
    return {"content": content}

def compare_performance_node(state: AgentState):
    content = "\n\n".join(state["content"] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is the financial analysis:\n\n{state['analysis']}"
    )
    messages = [
        SystemMessage(content=COMPETE_PERFORMANCE_PROMPT.format(content=content)),
        user_message,
    ]
    response = model.invoke(messages)
    return {
        "comparison": response.content,
        "revision_number": state.get("revision_number", 1) + 1,
    }
    
def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state["feedback"]),
        ]
    )
    content = state["content"] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response["results"]:
            content.append(r["content"])
    return {"content": content}


def collect_feedback_node(state: AgentState):
    messages = [
        SystemMessage(content=FEEDBACK_PROMPT),
        HumanMessage(content=state["comparison"]),
    ]
    response = model.invoke(messages)
    return {"feedback": response.content}

def write_report_node(state: AgentState):
    messages = [
        SystemMessage(content=WRITE_REPORT_PROMPT),
        HumanMessage(content=state["comparison"]),
    ]
    response = model.invoke(messages)
    return {"report": response.content}


def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "collect_feedback"

builder = StateGraph(AgentState)

builder.add_node("gather_financials", gather_financials_node)
builder.add_node("analyze_data", analyze_data_node)
builder.add_node("research_competitors", research_competitors_node)
builder.add_node("compare_performance", compare_performance_node)
builder.add_node("collect_feedback", collect_feedback_node)
builder.add_node("research_critique", research_critique_node)

builder.add_node("write_report", write_report_node)


builder.set_entry_point("gather_financials")


builder.add_conditional_edges(
    "compare_performance",
    should_continue,
    {END: END, "collect_feedback": "collect_feedback"},
)

builder.add_edge("gather_financials", "analyze_data")
builder.add_edge("analyze_data", "research_competitors")
builder.add_edge("research_competitors", "compare_performance")
builder.add_edge("collect_feedback", "research_critique")
builder.add_edge("research_critique", "compare_performance")
builder.add_edge("compare_performance", "write_report")

graph = builder.compile(checkpointer=memory)


# STREAMLIT FOR UI

import streamlit as st

def main():
    st.title("Financial Performance Reporting Agent")

    task = st.text_input(
        "Enter the task:",
        "Analyze the financial performance of our company (MyAICo.AI) compared to competitors",
    )
    competitors = st.text_area("Enter competitor names (one per line):").split("\n")
    max_revisions = st.number_input("Max Revisions", min_value=1, value=2)
    uploaded_file = st.file_uploader(
        "Upload a CSV file with the company's financial data", type=["csv"]
    )

    if st.button("Start Analysis") and uploaded_file is not None:
        # Read the uploaded CSV file
        csv_data = uploaded_file.getvalue().decode("utf-8")

        initial_state = {
            "task": task,
            "competitors": [comp.strip() for comp in competitors if comp.strip()],
            "csv_file": csv_data,
            "max_revisions": max_revisions,
            "revision_number": 1,
        }
        thread = {"configurable": {"thread_id": "1"}}

        with SqliteSaver.from_conn_string(":memory:") as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)

            final_state = None
            for s in graph.stream(initial_state, thread):
                st.write(s)
                final_state = s

            if final_state and "report" in final_state:
                st.subheader("Final Report")
                st.write(final_state["report"])


if __name__ == "__main__":
    main()