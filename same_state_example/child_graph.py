from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from .parent_graph import ParentState

load_dotenv()

class ParentState(TypedDict):
    question: str 
    answer_english: str 
    answer_hindi: str 

hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
)
subgraph_llm = ChatHuggingFace(llm=hf_endpoint)

# class SubgraphState(TypedDict):
#     input_text: str 
#     translated_text: str 

def translate_text_node(state: ParentState):
    prompt = f"""
    Translate the following text into Hindi language.
    Keep it natural and clear, Do not add extra content.

    Text: {state['answer_english']}
    """
    translated_text = subgraph_llm.invoke(prompt).content 
    return {'answer_hindi': translated_text}

subgraph_builder = StateGraph(ParentState)
subgraph_builder.add_node('translate_text_node', translate_text_node)
subgraph_builder.add_edge(START, 'translate_text_node')
subgraph_builder.add_edge('translate_text_node', END)

subgraph = subgraph_builder.compile()

