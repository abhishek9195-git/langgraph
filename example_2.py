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
import datetime, random

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# === Tools

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Simulates the latest stock price for a given ticker symbol (e.g., 'AAPL', 'TSLA').
    Returns a dictionary containing the symbol, current price, currency, and timestamp.
    """
    # Simulate a realistic price range based on common stocks
    price = round(random.uniform(10.0, 1000.0), 2)
    
    # Get current timestamp for 2025
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")    
    return {
        "symbol": symbol.upper(),
        "price": price,
        "currency": "USD",
        "last_updated": now,
        "status": "Success"
    }

@tool 
def purchase_stock(symbol: str, quantity: int) -> dict: 
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt and wait for a human decision (yes/ no).
    """

    decision = interrupt(f'Approve buying {quantity} shares of {symbol} ? (yes/ no)')
    if isinstance(decision, str) and decision.lower() == 'yes':
        return {
            'status': 'success',
            'message': f'Purchase order placed for {quantity} shares of {symbol}',
            'symbol': symbol,
            'quantity': quantity
        }
    else:
        return {
            'status': 'cancelled',
            'message': f'Purchase order cancelled for {quantity} shares of {symbol}',
            'symbol': symbol,
            'quantity': quantity
        }
    
tools = [get_stock_price, purchase_stock]
llm_with_tools = llm.bind_tools(tools)

# === State

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# === Nodes

def chat_node(state: ChatState):
    """
    LLM node that may answer or request a tool call.
    """
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages': response}

tool_node = ToolNode(tools)

# === Checkpointer

checkpointer = MemorySaver()

# === Graph

graph = StateGraph(ChatState)

# === Nodes

graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

# === Simple usages example (CLI WITH HITL)

if __name__ == '__main__':
    thread_id = 'thread01'
    CONFIG = {'configurable': {'thread_id': thread_id}}

    while True:
        user_input = input('You: ')
        if user_input.lower().strip() in {'exit', 'quit'}:
            print('Goodbye !')
            break 

        # Build initial state for this turn
        state = {'messages': [HumanMessage(content=user_input)]}

        # Run the graph (may hit an interrupt)
        result = chatbot.invoke(
            state,
            config=CONFIG
        )

        # Check for HITL interrupt from purchase stock
        interrupts = result.get('__interrupt__', [])

        if interrupts: # purchase_stock
            prompt_to_human = interrupts[0].value
            print(f'HITL: {prompt_to_human}')
            decision = input('Your decision: ').strip().lower()

            result = chatbot.invoke(
                Command(resume=decision),
                config=CONFIG
            )

        messages = result['messages']
        last_msg = messages[-1]
        print('==> last_msg', last_msg)

