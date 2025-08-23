# backend/agents/recipe_agent.py
import os, json, re
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Use the same OpenAI client you already use via AIML
from openai import OpenAI

load_dotenv()

# ---- One client for everything (AIML gateway compatible) ----
AIML_API_KEY  = os.getenv("AIML_API_KEY", "")
AIML_BASE_URL = os.getenv("AIML_BASE_URL", "https://api.aimlapi.com/v1")

# Let GPT-5 pick tools/approach for text; keep a small image model for image gen
TEXT_MODEL  = os.getenv("RECIPE_MODEL", "openai/gpt-5-chat-latest")
IMAGE_MODEL = os.getenv("RECIPE_IMAGE_MODEL", "openai/gpt-image-1")  

# Single client (no separate OpenAI key)
_client = OpenAI(base_url=AIML_BASE_URL, api_key=AIML_API_KEY)

SYSTEM_PROMPT_SUGGEST = """You are a culinary assistant.
Return JSON ONLY: an array like:
[{"name":"…","one_liner":"…"}]
Use the provided cuisine and ingredients. 5–7 ideas max. No preamble, just JSON.
"""

SYSTEM_PROMPT_DETAIL = """You are a concise recipe writer.
Return JSON ONLY with:
{
  "steps": ["short step 1", "..."],
  "servings": 2,
  "nutrition": {"kcal":..., "protein_g":..., "carbs_g":..., "fat_g":..., "fiber_g":..., "sodium_mg":...}
}
Keep 5–10 short steps. Per-serving nutrition. No preamble, just JSON.
"""

class RecipeState(dict):
    messages: List[BaseMessage]

def _extract_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None

def _text_llm():
    # Route text to GPT-5 chat via your AIML gateway
    return ChatOpenAI(
        model=TEXT_MODEL,
        api_key=AIML_API_KEY,
        base_url=AIML_BASE_URL,
        temperature=0.4,
        max_tokens=900,
    )

def _llm_node(state: RecipeState):
    llm = _text_llm()
    resp = llm.invoke(state["messages"])
    return {"messages": [resp]}

def build_recipe_graph():
    g = StateGraph(RecipeState)
    g.add_node("recipe_llm", _llm_node)
    g.set_entry_point("recipe_llm")
    g.add_edge("recipe_llm", END)
    return g.compile()

# Compile once
_recipe_graph = build_recipe_graph()

# ---- Image generation (same client via AIML) ----
def _maybe_generate_image(dish: str, cuisine: str) -> Dict[str, Any]:
    """
    Generates a food image using the AIML OpenAI-compatible images API.
    Returns {"image_url": "..."} or {"image_b64": "..."} or {} on failure.
    """
    # The AIML gateway must proxy the Images API. If not, we fail soft.
    prompt = (
        f"Professional food photography of '{dish}', {cuisine} style, plated appetizingly, "
        "natural light, shallow depth of field, high detail, no text."
    )
    try:
        img = _client.images.generate(
            model=IMAGE_MODEL,     # gpt-5-mini for images
            prompt=prompt,
            size="1024x1024",
            n=1
        )
        data0 = img.data[0]
        # Prefer URL if the gateway returns one, else b64
        if getattr(data0, "url", None):
            return {"image_url": data0.url}
        if getattr(data0, "b64_json", None):
            return {"image_b64": data0.b64_json}
    except Exception as e:
        # Don’t break the flow; bubble up a friendly note
        return {"image_error": f"Image generation unavailable: {e}"}
    return {}

# ---- Public helpers used by app_graph or main.py ----
def render_recipe_suggestions(payload_json: str) -> List[Dict[str, Any]]:
    """
    payload_json: {"ingredients":[...], "cuisine":"american", "count":5}
    Returns: [{"name":"...", "one_liner":"..."}, ...]
    """
    logger.info(f"recipe_agent.py DEBUG: render_recipe_suggestions")
    messages = [
        SystemMessage(content=SYSTEM_PROMPT_SUGGEST),
        HumanMessage(content=payload_json),
    ]
    result = _recipe_graph.invoke({"messages": messages})
    raw = result["messages"][-1].content or "[]"
    data = _extract_json(raw) or []
    req = json.loads(payload_json)
    count = int(req.get("count", 5))

    out = []
    for item in data[:count]:
        name = (item.get("name") or "").strip()
        one  = (item.get("one_liner") or "").strip()
        if name:
            out.append({"name": name, "one_liner": one})
    return out

def render_recipe_detail(payload_json: str) -> Dict[str, Any]:
    """
    payload_json: {"dish":"...", "ingredients":[...], "cuisine":"american"}
    Returns: {
      "steps":[...], "servings":int, "nutrition":{...},
      "image_url"|"image_b64"|"image_error": ...
    }
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT_DETAIL),
        HumanMessage(content=payload_json),
    ]
    result = _recipe_graph.invoke({"messages": messages})
    raw = result["messages"][-1].content or "{}"
    data = _extract_json(raw) or {}

    out: Dict[str, Any] = {
        "steps": data.get("steps") or [],
        "servings": data.get("servings") or 2,
        "nutrition": data.get("nutrition") or {},
    }

    # Generate image here (same client, GPT-5-mini)
    try:
        p = json.loads(payload_json)
        dish    = p.get("dish", "dish")
        cuisine = p.get("cuisine", "style")
    except Exception:
        dish, cuisine = "dish", "style"

    out.update(_maybe_generate_image(dish, cuisine))
    return out
