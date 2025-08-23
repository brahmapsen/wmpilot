# backend/agents/prediction_agent.py
import os
from typing import List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
AIML_API_KEY  = os.getenv("AIML_API_KEY", "")
AIML_BASE_URL = os.getenv("AIML_BASE_URL", "https://api.aimlapi.com/v1")
# Keep it flexible; you can set PREDICTION_MODEL in env. Defaults to a small, fast model.
MODEL = os.getenv("PREDICTION_MODEL", "openai/gpt-5-chat-latest")

SYSTEM_PROMPT = """You are a careful health coach. Write concise, evidence-aligned explanations.
Avoid medical advice. No guarantees. Use plain language. Return Markdown only. 120–180 words.
"""

class PredState(dict):
    messages: List[BaseMessage]

def _llm():
    logger.info("prediction_agent.py: _llm")
    return ChatOpenAI(
        model=MODEL,
        api_key=AIML_API_KEY,
        base_url=AIML_BASE_URL,
        temperature=0.2,
        max_tokens=600,
    )

def llm_node(state: PredState):
    logger.info("prediction_agent.py: llm_node")
    msgs = state.get("messages", [])
    resp = _llm().invoke(msgs)
    return {"messages": [resp]}

def build_prediction_graph():
    g = StateGraph(PredState)
    g.add_node("pred_llm", llm_node)
    g.set_entry_point("pred_llm")
    g.add_edge("pred_llm", END)
    return g.compile()
