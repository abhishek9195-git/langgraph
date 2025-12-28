from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List, Any
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import os
import requests

load_dotenv()

# Gemini 2.5 Flash-Lite is a valid stable model for 2025
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# --- Tools ---
search_tool = DuckDuckGoSearchRun(region='us-en')

@tool
def calculator(first_num: float, second_num: float, operation: str) -> float:
    """
    Calculates basic operation on two float numbers.
    Args:
        - first_num: float 
        - second_num: float
        - operation: add, sub, mul, div

    Returns:
        - Result of the operation: float type.

    Use this tool to perform calculations.
    """
    print(f'==> Tool call calculator first_num: {first_num}, second_num: {second_num}, operation: {operation}.')
    if operation == 'add': return first_num + second_num
    if operation == 'sub': return first_num - second_num
    if operation == 'mul': return first_num * second_num
    if operation == 'div': return first_num / second_num if second_num != 0 else "Error: Div by zero"
    return "Error: Invalid operation"

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
    print(f'Tool call getConvertedCurrency {url}')
    try:
        response = requests.get(url)
        data = response.json()
        if data.get('result') == 'success':
            return data['conversion_result']
    except Exception as e:
        return f"Error: {str(e)}"
    return None 

tools = [getConvertedCurrency, search_tool, calculator]
llm_with_tools = llm.bind_tools(tools)

# --- State ---
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# --- Nodes ---
def chat_node(state: ChatState):
    response = llm_with_tools.invoke(state['messages'])
    return {'messages': [response]}

tool_node = ToolNode(tools)

# --- Graph Structure ---
graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')

# Conditional edge: LLM -> Tools OR LLM -> END
graph.add_conditional_edges('chat_node', tools_condition)

# IMPORTANT: Edge to loop back to LLM after tool completes
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile()

# --- Execution ---
# if __name__ == "__main__":
    # Test with a tool-based query

# query = 'Convert 1 USD into INR. Then calculate 50 USD into INR.'
query = 'Multiply 1 with 10. Consider result as USD and convert it into INR.'
result = chatbot.invoke({'messages': [HumanMessage(content=query)]})
print(result['messages'][-1].content)
