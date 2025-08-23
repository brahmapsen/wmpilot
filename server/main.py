# backend/server/main.py
import os, json
from typing import Any, Dict, Optional, List
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from graph.app_graph import run_diet, run_exercise

from vapi_python import Vapi

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()



# Optional LangSmith tracing (set via env)
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=...
# LANGSMITH_PROJECT=WeightPilot
# (LangChain will auto-pick these up)

app = FastAPI(title="WeightPilot AI Agentic Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ---- Shared calcs ----
def bmi(height_cm: float, weight_kg: float) -> float:
    h_m = height_cm / 100.0
    return round(weight_kg / (h_m ** 2), 1)

def mifflin_st_jeor_bmr(sex_assigned_at_birth: str, weight_kg: float, height_cm: float, age: int) -> float:
    s = (sex_assigned_at_birth or "").lower()
    if s in ["male", "m"]:
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

def tdee_from_activity(bmr: float, activity_level: str) -> float:
    activity_map = {
        "sedentary": 1.2, "light": 1.375, "moderate": 1.55,
        "active": 1.725, "very active": 1.9,
    }
    return bmr * activity_map.get((activity_level or "").lower(), 1.375)

def compute_targets(profile: Dict[str, Any]) -> Dict[str, Any]:
    b = mifflin_st_jeor_bmr(profile["sex_assigned_at_birth"], profile["weight_kg"], profile["height_cm"], profile["age"])
    tdee = tdee_from_activity(b, profile["activity_level"])
    goal = (profile["goal"] or "maintain").lower()
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

# ---- Schemas ----
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

# class ExerciseRequest(BaseModel):
#     age: int
#     sex_assigned_at_birth: str
#     height_cm: float
#     weight_kg: float
#     activity_level: str
#     goal: str
#     constraints: Optional[List[str]] = []  # injuries, equipment limits, etc.

class ExerciseRequest(BaseModel):
    profile: Dict[str, Any]
    preferences: Optional[Dict[str, Any]] = None

class ExerciseResponse(BaseModel):
    plan_markdown: str

# ---- Routes ----
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/plan", response_model=PlanResponse)
def diet_plan(req: PlanRequest, x_api_key: Optional[str] = Header(default=None)):
    profile = req.dict()
    targets = compute_targets(profile)
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
        "calorie_target_hint": compute_targets(profile)["calorie_target"],
    }
    try:
        plan_md = run_diet(json.dumps(user_ctx, ensure_ascii=False), json.dumps(targets))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"diet agent error: {e}")
    if not plan_md.strip():
        raise HTTPException(status_code=502, detail="No content from diet agent")
    return {"targets": targets, "plan_markdown": plan_md}

@app.post("/v1/plan", response_model=PlanResponse)
def diet_plan(req: PlanRequest, x_api_key: Optional[str] = Header(default=None)):
    profile = req.dict()
    targets = compute_targets(profile)
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
        "calorie_target_hint": targets["calorie_target"],
    }
    try:
        plan_md = run_diet(json.dumps(user_ctx, ensure_ascii=False), json.dumps(targets))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"diet agent error: {e}")
    if not (plan_md or "").strip():
        raise HTTPException(status_code=502, detail="No content from diet agent")
    return {"targets": targets, "plan_markdown": plan_md}

@app.post("/v1/exercise/plan", response_model=ExerciseResponse)
def exercise_plan(req: ExerciseRequest):
    # Keep UI contract (profile + preferences), but mirror diet flow internally
    profile = req.profile or {}
    preferences = req.preferences or {}

    # Compute targets (safe defaults if some fields are missing)
    base = {
        "age": profile.get("age", 30),
        "sex_assigned_at_birth": profile.get("sex_assigned_at_birth", "male"),
        "height_cm": profile.get("height_cm", 170.0),
        "weight_kg": profile.get("weight_kg", 70.0),
        "activity_level": profile.get("activity_level", "moderate"),
        "goal": profile.get("goal", "maintain"),
        "goal_rate": profile.get("goal_rate", "moderate"),
    }
    try:
        targets = compute_targets(base)
    except Exception:
        targets = {}

    # User context for the exercise agent (parallel to diet)
    user_ctx = {
        **profile,
        "preferences": preferences,
        "calorie_target_hint": targets.get("calorie_target"),
    }

    logger.info(f"main.py DEBUG: created user ctx")

    try:
        plan_md = run_exercise(json.dumps(user_ctx, ensure_ascii=False), json.dumps(targets))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"exercise agent error: {e}")
    if not (plan_md or "").strip():
        raise HTTPException(status_code=502, detail="No content from exercise agent")
    return {"plan_markdown": plan_md}

#######################VAPI related functions################
from fastapi.responses import HTMLResponse

client_vapi = Vapi(api_key=os.getenv("VAPI_API_KEY"))

# Request/Response models
class CallRequest(BaseModel):
    assistant_id: str
    phone_number: str

class WebhookEvent(BaseModel):
    type: str
    call: dict = None
    transcript: str = None

# HTML template for the web interface
from string import Template

HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html>
<head>
  <title>Vapi FastAPI Example</title>
  <script src="https://cdn.jsdelivr.net/npm/@vapi-ai/web@latest/dist/vapi.js"></script>
</head>
<body>
  <h1>Vapi Voice Assistant (FastAPI)</h1>
  <button id="startCall">Start Call</button>
  <button id="endCall" disabled>End Call</button>
  <div id="status"></div>

  <script>
    const vapi = new Vapi("$vapi_public_key");
    const assistantId = "$assistant_id";

    document.getElementById('startCall').onclick = async () => {
      try {
        await vapi.start(assistantId);
        document.getElementById('startCall').disabled = true;
        document.getElementById('endCall').disabled = false;
        document.getElementById('status').innerText = 'Call started';
      } catch (error) { console.error('Error starting call:', error); }
    };

    document.getElementById('endCall').onclick = async () => {
      try {
        await vapi.stop();
        document.getElementById('startCall').disabled = false;
        document.getElementById('endCall').disabled = true;
        document.getElementById('status').innerText = 'Call ended';
      } catch (error) { console.error('Error ending call:', error); }
    };
  </script>
</body>
</html>""")

VAPI_PUBLIC_KEY = os.getenv("VAPI_PUBLIC_KEY")
VAPI_ASSISTANT_ID = os.getenv("VAPI_ASSISTANT_ID")

@app.get("/v1/vapi/config")
def vapi_config():
    if not VAPI_PUBLIC_KEY or not VAPI_ASSISTANT_ID:
        raise HTTPException(status_code=500, detail="VAPI_PUBLIC_KEY or VAPI_ASSISTANT_ID not set")
    return {"public_key": VAPI_PUBLIC_KEY, "assistant_id": VAPI_ASSISTANT_ID}

# @app.get("/vapi/widget", response_class=HTMLResponse)
# def vapi_widget():
#     if not VAPI_PUBLIC_KEY or not VAPI_ASSISTANT_ID:
#         raise HTTPException(status_code=500, detail="VAPI keys not configured")
#     html = HTML_TEMPLATE.substitute(vapi_public_key= VAPI_PUBLIC_KEY,assistant_id= VAPI_ASSISTANT_ID)
    
#     return HTMLResponse(content=html)

VAPI_WIDGET_HTML = r"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Vapi FastAPI Example</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; padding: 24px; }
    h1 { margin-bottom: 8px; }
    .row { margin: 10px 0; }
    #log { background:#0b1220; color:#d7e1f8; padding:12px; border-radius:8px; white-space:pre-wrap; }
    button { padding: 6px 10px; border-radius: 6px; border: 1px solid #d0d7de; background: #f6f8fa; cursor:pointer; }
    button:disabled { opacity: .6; cursor:not-allowed; }
  </style>
</head>
<body>
  <h1>Vapi Voice Assistant (FastAPI)</h1>

  <div class="row">
    <div>Assistant: <code>__ASSISTANT_ID__</code> · Key: <code>__PUBLIC_KEY_PART__</code></div>
  </div>

  <div class="row">
    <button id="startBtn">Start Call</button>
    <button id="stopBtn" disabled>End Call</button>
  </div>

  <div class="row">
    <div id="log">Status / Logs:</div>
  </div>

  <!-- Load the ESM build directly -->
  <script type="module">
    import Vapi from "https://cdn.jsdelivr.net/npm/@vapi-ai/web@latest/dist/vapi.js";

    const publicKey   = "__PUBLIC_KEY__";
    const assistantId = "__ASSISTANT_ID__";

    const log = (msg) => {
      const el = document.getElementById('log');
      el.textContent += "\\n" + msg;
    };

    let vapi;
    try {
      vapi = new Vapi(publicKey);
      log("Vapi SDK loaded.");
    } catch (e) {
      log("Failed to init Vapi: " + (e?.message || e));
    }

    const startBtn = document.getElementById('startBtn');
    const stopBtn  = document.getElementById('stopBtn');

    startBtn.onclick = async () => {
      try {
        await vapi.start(assistantId);
        startBtn.disabled = true;
        stopBtn.disabled = false;
        log("Call started. Grant microphone permission if prompted.");
      } catch (e) {
        log("Start error: " + (e?.message || e));
        console.error(e);
      }
    };

    stopBtn.onclick = async () => {
      try {
        await vapi.stop();
        startBtn.disabled = false;
        stopBtn.disabled = true;
        log("Call ended.");
      } catch (e) {
        log("Stop error: " + (e?.message || e));
        console.error(e);
      }
    };
  </script>
</body>
</html>
"""

@app.get("/vapi/widget", response_class=HTMLResponse)
async def vapi_widget():
    pub = os.getenv("VAPI_PUBLIC_KEY", "")
    asst = os.getenv("VAPI_ASSISTANT_ID", "")
    html = (
        VAPI_WIDGET_HTML
        .replace("__PUBLIC_KEY__", pub)
        .replace("__ASSISTANT_ID__", asst)
        .replace("__PUBLIC_KEY_PART__", (pub[:4] + "…" + pub[-4:]) if pub else "(none)")
    )
    return HTMLResponse(content=html)