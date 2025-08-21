# wm_app.py
import os
import json
import math
import time
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------
# Config (env-driven)
# -------------------------
load_dotenv()
USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"
AIML_API_KEY = os.getenv("AIML_API_KEY", "")
USDA_API_KEY = os.getenv("USDA_API_KEY")  # optional
NUTRITIONIX_APP_ID = os.getenv("NUTRITIONIX_APP_ID")  # optional
NUTRITIONIX_API_KEY = os.getenv("NUTRITIONIX_API_KEY")  # optional

MODEL = "openai/gpt-5-chat-latest"
# MODEL = "openai/gpt-5-2025-08-07"

if not AIML_API_KEY:
    raise RuntimeError("AIML_API_KEY is required")


# client = OpenAI(api_key=OPENAI_API_KEY)
client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key= AIML_API_KEY,
)

# -------------------------
# FastAPI app + CORS
# -------------------------
app = FastAPI(title="WeightPilot AI Engine", version="1.0.0")

# Allow localhost Streamlit by default; customize for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Helpers: calculations
# -------------------------
def bmi(height_cm: float, weight_kg: float) -> float:
    h_m = height_cm / 100
    return round(weight_kg / (h_m ** 2), 1)

def mifflin_st_jeor_bmr(sex_assigned_at_birth: str, weight_kg: float, height_cm: float, age: int) -> float:
    s = sex_assigned_at_birth.lower().strip()
    if s in ["male", "m"]:
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    # default to female if unspecified
    return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

def tdee_from_activity(bmr: float, activity_level: str) -> float:
    activity_map = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very active": 1.9,
    }
    factor = activity_map.get(activity_level.lower(), 1.375)
    return bmr * factor

def compute_targets(profile: Dict[str, Any]) -> Dict[str, Any]:
    b = mifflin_st_jeor_bmr(
        profile["sex_assigned_at_birth"], profile["weight_kg"], profile["height_cm"], profile["age"]
    )
    tdee = tdee_from_activity(b, profile["activity_level"])
    goal = profile["goal"]

    if goal == "lose":
        target = tdee * (0.8 if profile.get("goal_rate") == "aggressive" else 0.9)
    elif goal == "gain":
        target = tdee * (1.2 if profile.get("goal_rate") == "aggressive" else 1.1)
    else:
        target = tdee

    return {
        "bmi": bmi(profile["height_cm"], profile["weight_kg"]),
        "bmr": round(b, 0),
        "tdee": round(tdee, 0),
        "calorie_target": int(round(target, -1)),
    }

# -------------------------
# Optional external tools
# -------------------------
def usda_search_foods(query: str, page_size: int = 5):
    if not USDA_API_KEY:
        return {"error": "USDA_API_KEY not set"}
    url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {"api_key": USDA_API_KEY, "query": query, "pageSize": page_size}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def usda_food_details(fdc_id: int):
    if not USDA_API_KEY:
        return {"error": "USDA_API_KEY not set"}
    url = f"https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"
    params = {"api_key": USDA_API_KEY}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def nutritionix_parse(text: str):
    if not (NUTRITIONIX_APP_ID and NUTRITIONIX_API_KEY):
        return {"error": "NUTRITIONIX_APP_ID or NUTRITIONIX_API_KEY not set"}
    url = "https://trackapi.nutritionix.com/v2/natural/nutrients"
    headers = {
        "x-app-id": NUTRITIONIX_APP_ID,
        "x-app-key": NUTRITIONIX_API_KEY,
        "Content-Type": "application/json",
    }
    r = requests.post(url, headers=headers, json={"query": text}, timeout=20)
    r.raise_for_status()
    return r.json()

# Tool specs for GPT function-calling
FUNCTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "usda_search_foods",
            "description": "Search USDA FoodData Central for foods matching a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "page_size": {"type": "integer", "minimum": 1, "maximum": 25},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "usda_food_details",
            "description": "Fetch USDA FoodData Central details by FDC ID.",
            "parameters": {
                "type": "object",
                "properties": {"fdc_id": {"type": "integer"}},
                "required": ["fdc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nutritionix_parse",
            "description": "Parse free-text meals into nutrition facts using Nutritionix.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
]



PY_TOOL_IMPLS = {
    "usda_search_foods": usda_search_foods,
    "usda_food_details": usda_food_details,
    "nutritionix_parse": nutritionix_parse,
}

SYSTEM_PROMPT = """You are a nutrition & weight-management assistant for educational use.
- Use Mifflin-St Jeor for BMR, then an activity factor for TDEE.
- Goals: maintenance (±0), loss (-10% to -20%), gain (+10% to +20%).
- Default macros: 25–35% protein, 30–40% carbs, 25–35% fat unless user requests otherwise.
- Respect allergies, dietary restrictions, and cultural preferences. Do NOT change calories based on race/ethnicity; use them only to tailor menu ideas.
- Prefer whole foods, moderate added sugars, sodium ~≤2300 mg/day unless otherwise specified.
- You may call tools (USDA/Nutritionix) to check nutrition. If keys are missing, proceed without tools.
- Output: short intro, then a 7-day Markdown table (breakfast/lunch/dinner/snack) with approximate kcal/macros per meal and daily totals. End with a brief safety disclaimer.
"""

# -------------------------
# Pydantic models
# -------------------------
class PlanRequest(BaseModel):
    age: int
    sex_assigned_at_birth: str
    gender_identity: Optional[str] = None
    height_cm: float
    weight_kg: float
    activity_level: str
    goal: str
    goal_rate: str
    diet: Optional[str] = None
    allergies: List[str] = []
    medical_flags: List[str] = []
    cuisines: List[str] = []
    race_ethnicity: Optional[str] = None

class PlanResponse(BaseModel):
    targets: Dict[str, Any]
    plan_markdown: str = Field(..., description="Markdown plan")

# -------------------------
# GPT tool loop
# -------------------------

def run_response_with_tools(message: str, force_tool: str | None = None) -> str:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "usda_search_foods",
                "description": "Search USDA FoodData Central.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "page_size": {"type": "integer", "minimum": 1, "maximum": 25}
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    functions = [
        {
            "name": "usda_search_foods",
            "description": "Search USDA FoodData Central.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "page_size": {"type": "integer", "minimum": 1, "maximum": 25, "default": 5}
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    ]

    messages = [
        {"role": "system", "content": (
            SYSTEM_PROMPT +
            "\n\nYou have a function/tool named 'usda_search_foods'. "
            "When asked to check nutrition or when proposing meals, call it."
        )},
        {"role": "user", "content": message}
    ]

    # Encourage or force tool use
    tool_choice = {"type": "function", "function": {"name": force_tool}} if force_tool else "auto"


    resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=0.3,
            max_tokens=512,           # avoid finish_reason=length
        )
    msg = resp.choices[0].message
    logger.info(f"[tools] finish={resp.choices[0].finish_reason}; tool_calls={getattr(msg, 'tool_calls', None)}")

    if getattr(msg, "tool_calls", None):
        # 2) Execute tools, append results as `role="tool"` messages
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = usda_search_foods(**args)  # your python function
            logger.info(f"\n Appending to message: ")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result)
            })
        # 3) Ask model again to produce the final answer
        resp2 = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        final_text = resp2.choices[0].message.content
    else:
        logger.info(f"\n No tool calling: ")
        final_text = msg.content

    return final_text

def run_response_with_tools1(message: str) -> str:
    """
    Minimal tool loop:
      1) Call Responses API
      2) If there are tool calls, execute and send tool_outputs back
      3) Return final text
    """
    resp = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=message,
        # tools=FUNCTION_TOOLS,
        # verbosity="low",
        # reasoning={"effort": "medium"},
    )

    logger.info(f"\n First RESPONSE complete: ")

    # Execute at most a few tool rounds to avoid runaway loops in demos
    # for _ in range(3):
    #     tool_calls = [item for item in resp.output if getattr(item, "type", "") == "tool_call"]
    #     if not tool_calls:
    #         break

    #     tool_outputs = []
    #     for call in tool_calls:
    #         name = call.name
    #         args = call.input if hasattr(call, "input") else {}
    #         py_fn = PY_TOOL_IMPLS.get(name)
    #         try:
    #             result = py_fn(**args) if py_fn else {"error": f"Unknown tool {name}"}
    #         except Exception as e:
    #             result = {"error": str(e)}
    #         tool_outputs.append({"call_id": call.call_id, "output": json.dumps(result)})

    #     resp = client.responses.create(
    #         model=MODEL,
    #         instructions=SYSTEM_PROMPT,
    #         input=[{"role": "user", "content": message}],
    #         tools=FUNCTION_TOOLS,
    #         tool_outputs=tool_outputs,
    #         # verbosity="low",
    #         reasoning={"effort": "medium"},
    #     )

    return getattr(resp, "output_text", "") or ""

# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/v1/plan", response_model=PlanResponse)
def create_plan(req: PlanRequest, x_api_key: Optional[str] = Header(default=None)):
    """
    Streamlit calls this with the user's profile.
    Returns {"targets": {...}, "plan_markdown": "..."} as expected by the UI.
    """
    profile = req.dict()
    targets = compute_targets(profile)

    # Pre-brief the model with computed targets so it doesn't recompute differently.
    user_ctx = {
        "age": profile["age"],
        "sex_assigned_at_birth": profile["sex_assigned_at_birth"],
        "gender_identity": profile.get("gender_identity"),
        "height_cm": profile["height_cm"],
        "weight_kg": profile["weight_kg"],
        "activity_level": profile["activity_level"],
        "goal": profile["goal"],
        "goal_rate": profile.get("goal_rate"),
        "diet": profile.get("diet"),
        "allergies": profile.get("allergies", []),
        "medical_flags": profile.get("medical_flags", []),
        "cuisines": profile.get("cuisines", []),
        "race_ethnicity": profile.get("race_ethnicity"),
    }

    prompt = (
        "User profile (JSON):\n"
        f"{json.dumps(user_ctx, ensure_ascii=False)}\n\n"
        "Precomputed metrics (JSON):\n"
        f"{json.dumps(targets)}\n\n"
        "Task: Create a 7-day diet plan table grounded in typical foods the user will recognize. "
        "When appropriate, you may call tools to check nutrition for items. "
        "Return a concise intro paragraph, then the table as Markdown."
    )

    try:
        plan_md = run_response_with_tools(prompt, force_tool="usda_search_foods")
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Upstream HTTP error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    if not plan_md.strip():
        raise HTTPException(status_code=502, detail="No content returned from the model")

    return PlanResponse(targets=targets, plan_markdown=plan_md)

# -------------------------
# Optional: tool passthrough endpoints (for debugging)
# -------------------------
class USDAQuery(BaseModel):
    query: str
    page_size: int = 5

@app.post("/v1/tools/usda/search")
def api_usda_search(q: USDAQuery):
    return usda_search_foods(q.query, q.page_size)

@app.get("/v1/tools/usda/{fdc_id}")
def api_usda_food(fdc_id: int):
    return usda_food_details(fdc_id)

class NutriText(BaseModel):
    text: str

@app.post("/v1/tools/nutritionix/parse")
def api_nutritionix_parse(body: NutriText):
    return nutritionix_parse(body.text)

# -------------------------
# Local dev entrypoint
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("wm:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
