import os

import uvicorn
import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

model = ChatOpenAI(
    model="c1/openai/gpt-5/v-20250930",
    base_url="https://api.thesys.dev/v1/embed/",
    api_key=os.getenv("THESYS_API_KEY"),
)

# model = ChatOpenAI(
#     model=os.getenv("LLM_MODEL"),
#     base_url=os.getenv("LLM_BASE_URL"),
#     api_key=os.getenv("LLM_API_KEY"),
# )

checkpointer = InMemorySaver()
search = DuckDuckGoSearchRun()


def _get_stock(ticker: str) -> yf.Ticker | None:
    """
    Helper function to handle stock data
    """

    try:
        stock = yf.Ticker(ticker)
        return stock
    except Exception as e:
        return None


@tool(
    "get_internet_search_results",
    description="A function that searches the internet for the given query.",
)
def get_internet_search_results(query: str) -> str:
    """
    Use the internet to search for the given query and return the results.
    """
    result = search.run(query)
    return result


@tool(
    "get_stock_price",
    description="A function that returns the current stock price based on a ticker symbol.",
)
def get_stock_price(ticker: str):
    """
    Use the internet to search for the given query and return the results.
    """
    stock = _get_stock(ticker)

    if stock:
        return stock.history()["Close"].iloc[-1]
    return "Stock not found"


@tool(
    "get_historical_stock_price",
    description="A function that returns the current stock price over time based on a ticker symbol and a start and end date.",
)
def get_historical_stock_price(
    ticker: str, start_date: str = "2025-01-01", end_date: str = "2025-12-31"
):
    """
    Use the internet to search for the given query and return the results.
    """
    stock = _get_stock(ticker)

    if stock:
        return stock.history(start=start_date, end=end_date)["Close"]
    return "Stock not found"


@tool(
    "get_stock_news",
    description="A function that returns news based on a ticker symbol.",
)
def get_stock_news(ticker: str):
    """
    Use the internet to search for the given query and return the results.
    """
    stock = _get_stock(ticker)

    if stock:
        return stock.news
    return "Stock not found"


agent = create_agent(
    model=model,
    checkpointer=checkpointer,
    tools=[
        get_internet_search_results,
        get_stock_price,
        get_stock_news,
        get_historical_stock_price,
    ],
)


class PromptObject(BaseModel):
    content: str
    id: str
    role: str


class RequestObject(BaseModel):
    prompt: PromptObject
    threadId: str
    responseId: str


@app.post("/api/chat")
async def chat(request: RequestObject):
    config = {"configurable": {"thread_id": request.threadId}}

    def generate():
        for token, _ in agent.stream(
            {
                "messages": [
                    SystemMessage(
                        "You are a helpful assistant. You have solid knowledge of the stock market. And you have access to the internet. Your goal is to help the user by answering their questions."
                    ),
                    HumanMessage(request.prompt.content),
                ]
            },
            stream_mode="messages",
            config=config,
        ):
            yield token.content

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
