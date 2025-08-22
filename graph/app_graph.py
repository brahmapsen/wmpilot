# backend/graph/app_graph.py
from typing import Literal, Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage
from agents.diet_agent import build_diet_graph, SYSTEM_PROMPT as DIET_SYS
from agents.exercise_agent import build_exercise_graph, SYSTEM_PROMPT as EX_SYS

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Build graphs once (reuse across requests)
diet_graph = build_diet_graph()
exercise_graph = build_exercise_graph()

def run_diet(profile_json: str, targets_json: str) -> str:
    """Invoke the diet agent graph and return final assistant text."""
    messages = [
        SystemMessage(content=DIET_SYS),
        HumanMessage(content=(
            "User profile (JSON):\n" + profile_json +
            "\n\nPrecomputed metrics (JSON):\n" + targets_json +
            "\n\nTask: Create a 7-day diet plan table grounded in familiar foods."
        ))
    ]
    result = diet_graph.invoke({"messages": messages})
    # Last message should be assistant
    return result["messages"][-1].content


def run_exercise(profile_json: str, targets_json: str) -> str:
    messages = [
        SystemMessage(content=EX_SYS),
        HumanMessage(content=(
            "User profile (JSON):\n" + profile_json +
            "\n\nPrecomputed metrics (JSON):\n" + targets_json +
            "\n\nTask: Provide a weekly exercise regimen (cardio, strength, mobility). "
            "Call tools as needed to search exercises or estimate calorie burn. "
            "Output a Markdown table, then progression & safety notes."
        ))
    ]
    logger.info(f"app_grpah.py DEBUG: call exercise graph")
    result = exercise_graph.invoke({"messages": messages})
    return result["messages"][-1].content