import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from openai import OpenAI
import re
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

# pull your raw key...
raw_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not raw_key:
    raise ValueError("OPENAI_API_KEY not set")

# strip any ASCII or “smart” quotes
api_key = raw_key.strip().strip('"').strip("'").replace("“", "").replace("”", "")

#client = OpenAI(api_key=api_key)

llm_name="gpt-4o"

model = ChatOpenAI(api_key=api_key, model=llm_name)


# STEP 1: Build a basic chatbot
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    
tool = TavilySearchResults(max_results=2)
# resp = tool.invoke("Who is shouvik ahmed antu?")
# print(resp)
tools = [tool]

model_with_tools = model.bind_tools(tools)
    
def bot(state: State):
    print(state["messages"])
    return {"messages" : [model_with_tools.invoke(state["messages"])]}

graph_builder = StateGraph(State)

graph_builder.add_node("bot", bot)

graph_builder.set_entry_point("bot")

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges("bot", tools_condition)

#MEMORY
memory = SqliteSaver.from_conn_string(":memory:")


with SqliteSaver.from_conn_string(":memory:") as memory:
    graph = graph_builder.compile(checkpointer=memory)

    config = {
        "configurable" : {"thread_id": 1}
    }

    user_input = "Hi I'm bond. Do u know what fruit I like?"

    events = graph.stream({"messages": ("user", user_input)}, config, stream_mode="values")

    for event in events:
        event["messages"][-1].pretty_print()

# while True:
#     user_input = input("User: ")
#     if user_input in ["q", "quit", "exit"]:
#         print("goodbye!")
#         break
#     for event in graph.stream({"messages": ("user", user_input)}):
#         for value in event.values():
#             print("Assistant: ", value['messages'][-1].content)