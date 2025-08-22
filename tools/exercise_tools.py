# backend/tools/exercise_tools.py
import json
from typing import List, Dict, Any
from langchain_core.tools import tool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simple local library (swap to WGER/Strava/etc. later)
EXERCISE_LIBRARY = [
    {"name": "Brisk Walk",          "intensity": "light",     "kcal_per_min": 4},
    {"name": "Jogging",             "intensity": "moderate",  "kcal_per_min": 8},
    {"name": "Cycling (moderate)",  "intensity": "moderate",  "kcal_per_min": 7},
    {"name": "HIIT Circuit",        "intensity": "vigorous",  "kcal_per_min": 12},
    {"name": "Bodyweight Strength", "intensity": "moderate",  "kcal_per_min": 6},
]

@tool("search_exercises", return_direct=False)
def search_exercises(intensity: str = "moderate", limit: int = 3) -> str:
    """Return a few exercises matching an intensity (light/moderate/vigorous)."""
    logger.info(f"tools: search_exercises")
    out = [e for e in EXERCISE_LIBRARY if e["intensity"] == intensity][: max(1, min(limit, 5))]
    return json.dumps({"items": out}, ensure_ascii=False)

@tool("estimate_burn", return_direct=False)
def estimate_burn(name: str, minutes: int = 30, weight_kg: float = 70.0) -> str:
    """Crude calorie burn estimate (simple MET-like proxy)."""
    logger.info(f"tools: estimate_burn")
    base = next((e for e in EXERCISE_LIBRARY if e["name"].lower() == name.lower()), None)
    if not base:
        return json.dumps({"error": "exercise not found"})
    kcal = base["kcal_per_min"] * minutes * (weight_kg / 70.0)
    return json.dumps({"name": name, "minutes": minutes, "weight_kg": weight_kg, "kcal": round(kcal)})

