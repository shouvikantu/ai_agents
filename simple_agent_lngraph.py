import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from openai import OpenAI
import re
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

load_dotenv()

# pull your raw key...
raw_key = os.getenv("OPENAI_API_KEY")
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
    
def bot(state: State):
    print(state["messages"])
    return {"messages" : [model.invoke(state["messages"])]}

graph_builder = StateGraph(State)

graph_builder.add_node("bot", bot)

graph_builder.set_entry_point("bot")

graph_builder.set_finish_point("bot")

graph = graph_builder.compile()

# res = graph.invoke({"messages": ["Hello, How are you?"]})

# print(res)

while True:
    user_input = input("User: ")
    if user_input in ["q", "quit", "exit"]:
        print("goodbye!")
        break
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            print("Assistant: ", value['messages'][-1].content)