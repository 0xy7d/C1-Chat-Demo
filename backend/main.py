import os
from dotenv import load_dotenv
from pydantic import BaseModel

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.tools import DuckDuckGoSearchRun

import yfinance as yf

load_dotenv()

app = FastAPI()

model = ChatOpenAI(
    model = 'c1/openai/gpt-5/v-20250930',
    base_url = 'https://api.thesys.dev/v1/embed/'
)

# model = ChatOpenAI(
#     model = './qwen2.5-7b',
#     base_url = os.getenv('AI_API_BASE_URL'),
#     api_key = os.getenv('AI_API_KEY')
# )

checkpointer = InMemorySaver()
search = DuckDuckGoSearchRun()

@tool('get_internet_search_results', description='A function that searches the internet for the given query.')
def get_internet_search_results(query: str) -> str:
    """
    Use the internet to search for the given query and return the results.
    """
    result = search.run(query)
    return result

@tool('get_stock_price', description='A function that returns the current stock price based on a ticker symbol.')
def get_stock_price(ticker: str):
    """
    Use the internet to search for the given query and return the results.
    """
    stock = yf.Ticker(ticker)
    return stock.history()['Close'].iloc[-1]

@tool('get_historical_stock_price', description='A function that returns the current stock price over time based on a ticker symbol and a start and end date.')
def get_historical_stock_price(ticker: str, start_date: str, end_date: str):
    """
    Use the internet to search for the given query and return the results.
    """
    stock = yf.Ticker(ticker)
    return stock.history(start=start_date, end=end_date)['Close']

@tool('get_stock_news', description='A function that returns news based on a ticker symbol.')
def get_stock_news(ticker: str):
    """
    Use the internet to search for the given query and return the results.
    """
    stock = yf.Ticker(ticker)
    return stock.news

agent = create_agent(
    model = model,
    checkpointer = checkpointer,
)


class PromptObject(BaseModel):
    content: str
    id: str
    role: str


class RequestObject(BaseModel):
    prompt: PromptObject
    threadId: str
    responseId: str


@app.post('/api/chat')
async def chat(request: RequestObject):
    config = {'configurable': {'thread_id': request.threadId}}

    def generate():
        for token, _ in agent.stream(
            {'messages': [
                SystemMessage(''),
                HumanMessage(request.prompt.content)
            ]},
            stream_mode='messages',
            config=config
        ):
            yield token.content

    return StreamingResponse(generate(), media_type='text/event-stream',
                             headers={
                                 'Cache-Control': 'no-cache, no-transform',
                                 'Connection': 'keep-alive',
                             })

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
