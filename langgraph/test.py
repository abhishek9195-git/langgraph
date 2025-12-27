from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')

# === Schema 

class SentimentSchema(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Sentiment of the reivew.')

structured_model = model.with_structured_output(SentimentSchema)

class ReviewState(TypedDict):
    review: str
    sentiment: Literal['positive', 'negative']
    diagnosis: dict
    response: str

def find_sentiment(state: ReviewState):
    prompt = f'For the following review find out the sentiment \n {state["review"]}'
    sentiment = structured_model.invoke(prompt).sentiment
    print('===> sentiment', sentiment)
    state['sentiment'] = sentiment
    return {sentiment: 'sentiment'}


graph = StateGraph(ReviewState)
graph.add_node('find_sentiment', find_sentiment)

graph.add_edge(START, 'find_sentiment')
graph.add_edge('find_sentiment', END)

workflow = graph.compile()

initialState = {
    'review': 'This software hangs a lot.'
}

workflow.invoke(initialState)
print(workflow)

