from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Literal, Any, Annotated, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import operator
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
import time

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# node function
def chat_node(state: ChatState):
    messages = state['messages']
    response =llm.invoke(messages)
    return {'messages': [response]}

# checkpointer
checkpointer = InMemorySaver()

# node
graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

