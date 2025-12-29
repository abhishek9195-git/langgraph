from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from child_graph import subgraph
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# parent_graph_llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
hf_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
)
parent_graph_llm = ChatHuggingFace(llm=hf_endpoint)
subgraph_llm = ChatHuggingFace(llm=hf_endpoint)


class ParentState(TypedDict):
    question: str 
    answer_english: str 
    answer_hindi: str 

# def translate_text_node(state: ParentState):
#     prompt = f"""
#     Translate the following text into Hindi language.
#     Keep it natural and clear, Do not add extra content.

#     Text: {state['answer_english']}
#     """
#     translated_text = subgraph_llm.invoke(prompt).content 
#     return {'answer_hindi': translated_text}

# subgraph_builder = StateGraph(ParentState)
# subgraph_builder.add_node('translate_text_node', translate_text_node)
# subgraph_builder.add_edge(START, 'translate_text_node')
# subgraph_builder.add_edge('translate_text_node', END)

# subgraph = subgraph_builder.compile()

def generate_answer(state: ParentState):
    prompt = f"""
    You are a helpful assistant. Answer clearly.
    Question: {state['question']}
    """

    response = parent_graph_llm.invoke(prompt).content
    return {'answer_english': response}

# def translate_answer(state: ParentState):
#     input: SubgraphState = {
#         'input_text': state['answer_english']
#     }
#     subgraph_response: SubgraphState = subgraph.invoke(input)
#     return {
#         'answer_hindi': subgraph_response['translated_text']
#     }

parent_builder = StateGraph(ParentState)

parent_builder.add_node('answer', generate_answer)
parent_builder.add_node('translate', subgraph) # subgraph as node

parent_builder.add_edge(START, 'answer')
parent_builder.add_edge('answer', 'translate')
parent_builder.add_edge('translate', END)

workflow = parent_builder.compile()

response = workflow.invoke({'question': 'Machine learning in less than 200 words.'})

print('==> ', response)
