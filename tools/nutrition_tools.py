# backend/tools/nutrition_tools.py
import os
import json
import requests
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_core.tools import tool

load_dotenv()
USDA_API_KEY = os.getenv("USDA_API_KEY")
NUTRITIONIX_APP_ID = os.getenv("NUTRITIONIX_APP_ID")
NUTRITIONIX_API_KEY = os.getenv("NUTRITIONIX_API_KEY")

KEY_NUTRIENTS = {
    208: "kcal",        # Energy (KCAL)
    203: "protein_g",   # Protein
    205: "carbs_g",     # Carbohydrate
    204: "fat_g",       # Total fat
    291: "fiber_g",     # Fiber
    307: "sodium_mg",   # Sodium
}

def _extract_macros_from_nutrients(nutr_list):
    out = {v: None for v in KEY_NUTRIENTS.values()}
    for fn in (nutr_list or []):
        nid = fn.get("nutrientId")
        if nid in KEY_NUTRIENTS:
            out[KEY_NUTRIENTS[nid]] = fn.get("value")
    return out

def _compact_search(payload: dict, top_k: int = 3) -> dict:
    foods = (payload or {}).get("foods", [])[:top_k]
    compact = []
    for f in foods:
        compact.append({
            "fdcId": f.get("fdcId"),
            "description": f.get("description"),
            "dataType": f.get("dataType"),
            "publishedDate": f.get("publishedDate"),
            "category": f.get("foodCategory"),
            "brandOwner": f.get("brandOwner"),
            **_extract_macros_from_nutrients(f.get("foodNutrients")),
        })
    return {
        "totalHits": (payload or {}).get("totalHits"),
        "returned": len(compact),
        "items": compact,
    }

def _compact_details(food: dict) -> dict:
    portions = []
    for p in (food or {}).get("foodPortions") or []:
        if len(portions) >= 2:
            break
        portions.append({
            "mass_g": p.get("gramWeight"),
            "measure": p.get("portionDescription") or p.get("modifier"),
        })
    return {
        "fdcId": food.get("fdcId"),
        "description": food.get("description"),
        "dataType": food.get("dataType"),
        "brandOwner": food.get("brandOwner"),
        "servingSize": food.get("servingSize"),
        "servingSizeUnit": food.get("servingSizeUnit"),
        "macros": _extract_macros_from_nutrients(food.get("foodNutrients")),
        "portions": portions,
    }

@tool("usda_search_foods", return_direct=False)
def usda_search_foods(query: str, page_size: int = 3) -> str:
    """Search USDA FoodData Central for a food query.
    Returns a compact JSON string with top results and macros."""
    if not USDA_API_KEY:
        return json.dumps({"error": "USDA_API_KEY not set"})
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {"api_key": USDA_API_KEY, "query": query, "pageSize": max(1, min(page_size, 5))}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    compact = _compact_search(r.json(), top_k=params["pageSize"])
    return json.dumps(compact, ensure_ascii=False)

@tool("usda_food_details", return_direct=False)
def usda_food_details(fdc_id: int) -> str:
    """Fetch USDA FoodData Central details by FDC ID.
    Returns compact JSON string with macros and a couple of portions."""
    if not USDA_API_KEY:
        return json.dumps({"error": "USDA_API_KEY not set"})
    url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"
    params = {"api_key": USDA_API_KEY}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    compact = _compact_details(r.json())
    return json.dumps(compact, ensure_ascii=False)

@tool("nutritionix_parse", return_direct=False)
def nutritionix_parse(text: str) -> str:
    """Parse free-text foods into nutrition facts using Nutritionix."""
    if not (NUTRITIONIX_APP_ID and NUTRITIONIX_API_KEY):
        return json.dumps({"error": "NUTRITIONIX_APP_ID or NUTRITIONIX_API_KEY not set"})
    url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
    headers = {
        "x-app-id": NUTRITIONIX_APP_ID,
        "x-app-key": NUTRITIONIX_API_KEY,
        "Content-Type": "application/json",
    }
    r = requests.post(url, headers=headers, json={"query": text}, timeout=20)
    r.raise_for_status()
    return json.dumps(r.json(), ensure_ascii=False)
