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

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

# === Load document
loader = PyPDFLoader('introduction_to_ml.pdf')
docs = loader.load()


# === Split the text
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)


# === Generate embedding of each chunk

model_name = "all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'} # Use 'cuda' if you have a GPU
encode_kwargs = {'normalize_embeddings': False}

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model
)

# === Retriever

retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs={'k': 4})

# === Wrap the retriever inside tool

@tool 
def rag_tool(query: str):
    """
    Retrieve relevant information from the PDF document.
    Use this tool when the user asks factual/ conceptual questions that might be answered from the stored documents.
    """
    print('[Tool] rag tool invoked.')
    relevant_documents = retriever.invoke(query)

    context = [doc.page_content for doc in relevant_documents]
    metadata = [doc.metadata for doc in relevant_documents]

    return {
        'query': query,
        'context': context,
        'metadata': metadata
    }


# === tools

tools = [rag_tool]

llm_with_tools = llm.bind_tools(tools)


# --- Langgraph begins---

# === State

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# === Nodes

def chat_node(state: ChatState):
    response = llm_with_tools.invoke(state['messages'])
    return {'messages': [response]}

tool_node = ToolNode(tools)

# === Graph

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node') 

chatbot = graph.compile()

# === Invoke

chatbot_response = chatbot.invoke({'messages': [HumanMessage(content='Why machine learning ?')]})

print('==> chatbot_response', chatbot_response)
