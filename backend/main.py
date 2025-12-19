import logging
import os

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
logger = logging.getLogger(__name__)

origins = [
    "http://localhost:3000",
    "https://chat.demo.0xy7d.xyz",
    "https://chat-demo.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


def _get_stock(ticker: str) -> yf.Ticker:
    return yf.Ticker(ticker)


@tool(
    "get_internet_search_results",
    description="Search the internet for the given query.",
)
def get_internet_search_results(query: str):
    try:
        return search.run(query)
    except Exception as e:
        logger.exception("internet_search_failed")
        return {"error": "internet_search_failed", "details": str(e)}


@tool(
    "get_stock_price",
    description="Returns the latest closing stock price for a ticker symbol.",
)
def get_stock_price(ticker: str):
    try:
        stock = _get_stock(ticker)
        history = stock.history(period="1d")
        if history.empty:
            return {"error": "no_price_data"}
        return {
            "ticker": ticker.upper(),
            "price": float(history["Close"].iloc[-1]),
        }
    except Exception as e:
        logger.exception("stock_price_failed")
        return {"error": "stock_price_failed", "details": str(e)}


@tool(
    "get_historical_stock_price",
    description="Returns historical closing prices for a ticker symbol.",
)
def get_historical_stock_price(
    ticker: str,
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
):
    try:
        stock = _get_stock(ticker)
        history = stock.history(start=start_date, end=end_date)
        if history.empty:
            return {"error": "no_historical_data"}
        return {
            "ticker": ticker.upper(),
            "prices": history["Close"].to_dict(),
        }
    except Exception as e:
        logger.exception("historical_price_failed")
        return {"error": "historical_price_failed", "details": str(e)}


@tool(
    "get_stock_news",
    description="Returns recent news articles for a ticker symbol.",
)
def get_stock_news(ticker: str):
    try:
        stock = _get_stock(ticker)
        news = stock.news
        if not news:
            return {"error": "no_news"}
        return {
            "ticker": ticker.upper(),
            "news": news,
        }
    except Exception as e:
        logger.exception("stock_news_failed")
        return {"error": "stock_news_failed", "details": str(e)}


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
                        "You are a helpful assistant with strong knowledge of the stock market and access to the internet."
                    ),
                    HumanMessage(request.prompt.content),
                ]
            },
            stream_mode="messages",
            config=config,
        ):
            if token.content:
                yield token.content

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
        },
    )

