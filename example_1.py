from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Any, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode, tools_condition
import os
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

# === State

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# === Nodes

def chat_node(state: ChatState):
    decision = interrupt({
        'type': 'approval',
        'reason': 'Model is about to answer a user question.',
        'question': state['messages'][-1].content,
        'instruction': 'Approve this question (yes/ no)'
    })
    if decision['approved'] == 'no':
        return {'messages': [AIMessage(content='Not approved.')]}
    else:
        response = llm.invoke(state['messages'])
        return {'messages': [response]}


# === Graph

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END) 


# === Checkpointer (required for interrupts)

checkpointer = MemorySaver()

chatbot = graph.compile(checkpointer=checkpointer)

CONFIG = {'configurable': {'thread_id': '1'}}

# === Invoke

chatbot_approval = chatbot.invoke({'messages': [HumanMessage(content='Who is peaky blinders')]}, config=CONFIG)

confirmation_obj = chatbot_approval['__interrupt__'][0].value
user_input = input(f'\n Backend message - {confirmation_obj} \n Approve this question (yes/no)')

final_result = chatbot.invoke(Command(resume={'approved': user_input}), config=CONFIG)

print('==> final_result', final_result)
