# backend/graph/app_graph.py
import os, json, re
from typing import Literal, Dict, Any, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from agents.diet_agent import build_diet_graph, SYSTEM_PROMPT as DIET_SYS
from agents.exercise_agent import build_exercise_graph, SYSTEM_PROMPT as EX_SYS
from agents.coach_agent import build_coach_graph, SYSTEM_PROMPT as COACH_SYS

from agents.prediction_agent import build_prediction_graph, SYSTEM_PROMPT as PRED_SYS

from agents.recipe_agent import (
    render_recipe_suggestions as _recipe_suggestions,
    render_recipe_detail as _recipe_detail,
)

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Build graphs once (reuse across requests)
diet_graph = build_diet_graph()
exercise_graph = build_exercise_graph()

coach_graph = build_coach_graph()
prediction_graph = build_prediction_graph()

def run_diet(profile_json: str, targets_json: str) -> Dict[str, Any]:
    messages = [
        SystemMessage(content=DIET_SYS),
        HumanMessage(content=(
            "User profile (JSON):\n" + profile_json +
            "\n\nPrecomputed metrics (JSON):\n" + targets_json +
            "\n\nTask: Create a 7-day diet plan table grounded in familiar foods.\n"
            "Also return a short JSON object with keys:\n"
            "{'why_summary': str (<=80 words),\n"
            " 'personalization': [str],\n"
            " 'assumptions': [str],\n"
            " 'safety_checks': [str] }\n"
            "Do not include private chain-of-thought; just the final concise rationale."
        ))
    ]
    result = diet_graph.invoke({"messages": messages})
    txt = result["messages"][-1].content

    # Expect two blocks: the Markdown plan + a JSON blob after '---\nEXPLAIN:\n{...}'
    why = {"why_summary":"", "personalization":[], "assumptions":[], "safety_checks":[]}
    plan_md = txt
    if "\n---\nEXPLAIN:\n" in txt:
        plan_md, explain_json = txt.split("\n---\nEXPLAIN:\n", 1)
        import json as _json
        try:
            why = _json.loads(explain_json.strip())
        except Exception:
            pass
    return {"plan_markdown": plan_md.strip(), "why": why}

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


def run_recipe_suggest(ingredients: List[str], cuisine: str, count: int = 5) -> List[Dict[str, Any]]:
    """
    Return a small list of dishes:
    [{"name": "...", "one_liner": "..."}]
    """
    logger.info(f"app_grpah.py DEBUG: run_recipe_suggest")
    payload = {
        "ingredients": ingredients or [],
        "cuisine": (cuisine or "").strip() or "american",
        "count": max(1, min(count, 7)),
    }
    return _recipe_suggestions(json.dumps(payload, ensure_ascii=False))

def run_recipe_detail(dish: str, ingredients: List[str], cuisine: str) -> Dict[str, Any]:
    """
    Return detailed recipe with steps, per-serving nutrition, and (if available) image info:
    {
      "steps": [...],
      "servings": int,
      "nutrition": {...},
      "image_url" | "image_b64" | "image_error": ...
    }
    """
    payload = {
        "dish": (dish or "").strip(),
        "ingredients": ingredients or [],
        "cuisine": (cuisine or "").strip() or "american",
    }
    return _recipe_detail(json.dumps(payload, ensure_ascii=False))

## Find coaches
def run_coach_search(
    zip: Optional[str],
    city: Optional[str],
    state: Optional[str],
    roles: List[str],
    radius_km: int,
    max_results: int,
) -> List[Dict[str, Any]]:
    """
    Returns a list[dict] of professionals:
    {name, role, rating, reviews, address, phone, website, link, distance_km?}
    """

    logger.info("INFO: app_graph: run_coach_search() ")
    user_payload = {
        "zip": zip,
        "city": city,
        "state": state,
        "roles": roles,
        "radius_km": radius_km,
        "max_results": max_results,
    }

    messages = [
        SystemMessage(content=COACH_SYS),
        HumanMessage(content=(
            "Find local professionals based on this JSON:\n" +
            json.dumps(user_payload, ensure_ascii=False)
        )),
    ]
    roles = [m.__class__.__name__ for m in messages]
    logger.info(f"run_coach_search -> sending roles:{roles}")

    result = coach_graph.invoke({"messages": messages})
    content = result["messages"][-1].content or ""

    # Expect a JSON list in the final assistant message
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "items" in data:
            return data["items"]
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # Fallback: if the tool returned JSON directly in a previous tool message,
    # the model may have just echoed text. You can make this stricter if you like.
    return []

# NEW: moved from LLM inline to an agent call
def run_progress_explainer(profile_json: str, assumptions_json: str) -> str:
    messages = [
        SystemMessage(content=PRED_SYS),
        HumanMessage(content=(
            "Context:\n"
            f"User profile JSON:\n{profile_json}\n\n"
            f"Simulation assumptions JSON:\n{assumptions_json}\n\n"
            "Task: In 120–180 words, explain:\n"
            "• Why this calorie target plus activity usually leads to gradual weight loss.\n"
            "• Expected pace (early vs later) and normal plateaus.\n"
            "• What factors can speed/slow progress.\n"
            "• One-line safety note.\n"
            "Return Markdown only."
        ))
    ]
    result = prediction_graph.invoke({"messages": messages})
    return result["messages"][-1].content