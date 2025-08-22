# backend/agents/exercise_agent.py
import os
from typing import List, TypedDict, Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

from tools.exercise_tools import search_exercises, estimate_burn

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
AIML_API_KEY = os.getenv("AIML_API_KEY", "")
AIML_BASE_URL = os.getenv("AIML_BASE_URL", "https://api.aimlapi.com/v1")
MODEL = os.getenv("EXERCISE_MODEL", "openai/gpt-4o")

SYSTEM_PROMPT = """You are an evidence-informed exercise coach.
- Create weekly routines aligned with user goals, time availability, equipment, and medical flags.
- Balance cardio, strength, and mobility. Prioritize safety and progressive overload.
- You can call tools to suggest exercises and estimate calorie burn.
- Output: a clear weekly plan (Markdown table), then quick tips and safety notes.
"""

# --- State with proper message aggregation (mirrors diet agent pattern) ---
class ExerciseState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def exercise_llm():
    logger.info(f"exercise_agent.py:exercise_llm: ")
    if not AIML_API_KEY:
        raise RuntimeError("AIML_API_KEY is required")
    return ChatOpenAI(
        model=MODEL,
        api_key=AIML_API_KEY,
        base_url=AIML_BASE_URL,
        temperature=0.2,
        max_tokens=1000,
    ).bind_tools([search_exercises, estimate_burn])

tool_node = ToolNode([search_exercises, estimate_burn])

def llm_node(state: ExerciseState):
    logger.info(f"exercise_agent.py:llm-node")
    llm = exercise_llm()
    resp = llm.invoke(state["messages"])
    return {"messages": [resp]}

def build_exercise_graph():
    logger.info(f"build_exercise_graph: DEBUG: created user ctx")
    g = StateGraph(ExerciseState)
    g.add_node("exercise_llm", llm_node)
    g.add_node("tools", tool_node)

    g.set_entry_point("exercise_llm")
    g.add_conditional_edges("exercise_llm", tools_condition, {
        "tools": "tools",
        END: END,
    })
    g.add_edge("tools", "exercise_llm")
    return g.compile()

