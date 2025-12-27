from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Any, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import sqlite3

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

connection = sqlite3.connect(database='chatbot.db', check_same_thread=False)
check_pointer = SqliteSaver(conn=connection)

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot_workflow = graph.compile(checkpointer=check_pointer)

CONFIG = {'configurable': {'thread_id': 'thread_1'}}

def retrive_all_threads():
    all_threads = set()

    for checkpoint in check_pointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)
