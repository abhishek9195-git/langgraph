import asyncio
import os
import sys
from pathlib import Path
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

mcp_server_path = str(Path(__file__).parent.absolute() / "mcp_server.py")

# Use sys.executable to ensure the server uses the SAME venv as your main script
executable = sys.executable 

client = MultiServerMCPClient(
    {
        'arith': {
            'transport': 'stdio',
            'command': executable, 
            'args': [mcp_server_path],
            'env': os.environ.copy() # IMPORTANT: Server needs your PATH/Env to run
        },
        # 'expense': {
        #     'transport': 'streamable_http', 
        #     'url': 'https://splendid-gold-dingo.fastmcp.app/mcp'
        # }
    }
)

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

async def build_graph():
    try:

        tools = await asyncio.wait_for(client.get_tools(), timeout=10.0)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        llm_with_mcp_tools = llm.bind_tools(tools)

        async def chat_node(state: ChatState):
            response = await llm_with_mcp_tools.ainvoke(state["messages"])
            return {"messages": [response]}
        
        tool_node = ToolNode(tools)

        graph = StateGraph(ChatState)

        graph.add_node("chat_node", chat_node)
        graph.add_node("tools", tool_node)
        graph.add_edge(START, "chat_node")
        graph.add_conditional_edges("chat_node", tools_condition)
        graph.add_edge("tools", "chat_node")
        return graph.compile()
    
    except asyncio.TimeoutError:
        print("CRITICAL ERROR: MCP Server failed to respond within 10 seconds.")
    except Exception as e:
        print(f"Error inside build_graph: {e}")
        return None

async def main():
    try:
        chatbot = await build_graph()
        query = "Multiply 1 with 10"
        response = await chatbot.ainvoke({"messages": [HumanMessage(content=query)]})
        print(response['messages'])
    except Exception as e:
        print(f"Graph Error: {e}")

if __name__ == '__main__':
    asyncio.run(main())