from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Any, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import sqlite3
import requests
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
import os
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

# tools
search_tool = DuckDuckGoSearchRun(region = 'us-en')

@tool
def getConvertedCurrency(from_currency: str, to_currency: str, amount: float) -> Any:
    """ 
    Currency converter tool. Fetches the converted value from an API.
    Args:
        from_currency: The base currency code (e.g., 'USD')
        to_currency: The target currency code (e.g., 'EUR')
        amount: The numerical amount to convert
    """
    api_key = os.environ.get('EXCHANGE_RATE_API_KEY')
    url = f'https://v6.exchangerate-api.com/v6/{api_key}/pair/{from_currency}/{to_currency}/{amount}'
    print(f'[Tool] /getConvertedCurrency {url}')
    try:
        response = requests.get(url)
        data = response.json()
        if data.get('result') == 'success':
            return data['conversion_result']
    except Exception as e:
        return f"Error: {str(e)}"
    return None 

tools = [search_tool, getConvertedCurrency]
llm_with_tools = llm.bind_tools(tools)


# === State

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# === Nodes

def chat_node(state: ChatState):
    response = llm_with_tools.invoke(state['messages'])
    return {'messages': [response]}

tool_node = ToolNode(tools)

# === checkpointer

connection = sqlite3.connect(database='chatbot.db', check_same_thread=False)
check_pointer = SqliteSaver(conn=connection)

# === Graph

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot_workflow = graph.compile(checkpointer=check_pointer)

CONFIG = {'configurable': {'thread_id': 'thread_1'}}

def retrive_all_threads():
    all_threads = set()

    for checkpoint in check_pointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)