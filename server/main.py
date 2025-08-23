# backend/server/main.py
import os, json
from typing import Any, Dict, Optional, List
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from graph.app_graph import run_diet, run_exercise, run_recipe_suggest, run_recipe_detail, run_coach_search

from vapi_python import Vapi

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from datetime import date, timedelta

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

class ExerciseRequest(BaseModel):
    profile: Dict[str, Any]
    preferences: Optional[Dict[str, Any]] = None

class ExerciseResponse(BaseModel):
    plan_markdown: str

####  TRUST FACTOR RELATED schemas
class Source(BaseModel):
    title: str
    url: str

class ForecastPoint(BaseModel):
    day: int
    date: str
    weight_kg: float
    p10: float
    p90: float

class PlanTrust(BaseModel):
    why_summary: str
    personalization: List[str]
    assumptions: List[str]
    safety_checks: List[str]
    provenance: Dict[str, Any]          # e.g., {"usda": [{"fdcId":..., "desc":...}], "tools_used":[...]}
    daily_math: Dict[str, Any]           # e.g., {"kcal_per_meal":[...], "totals":[...]}
    forecast: List[ForecastPoint]
    citations: List[Source]

class PlanResponse(BaseModel):
    targets: Dict[str, Any]
    plan_markdown: str
    trust: PlanTrust

# ---- Recipe Schemas ----
class RecipeSuggestRequest(BaseModel):
    ingredients: List[str]
    cuisine: str = "american"
    count: int = 5

class RecipeSuggestResponse(BaseModel):
    suggestions: List[Dict[str, Any]]

class RecipeDetailRequest(BaseModel):
    dish: str
    ingredients: List[str] = []
    cuisine: str = "american"

class RecipeDetailResponse(BaseModel):
    dish: str
    steps: List[str]
    servings: int
    nutrition: Dict[str, Any]
    image_url: Optional[str] = None
    image_b64: Optional[str] = None
    image_error: Optional[str] = None


# --- rough-but-honest forecast helper (12 weeks, with uncertainty band) ---
def forecast_weight(weight_kg: float, tdee: float, calorie_target: float, weeks: int = 12) -> List[ForecastPoint]:
    # Expected loss uses a simple energy balance (≈7700 kcal per kg).
    # This is an approximation; we show it transparently and bound it with uncertainty.
    deficit = max(0.0, tdee - calorie_target)
    daily_delta_kg = deficit / 7700.0
    pts = []
    w = weight_kg
    start = date.today()
    for wk in range(weeks + 1):
        if wk > 0:
            w = max(0.0, w - (daily_delta_kg * 7.0))
        # Add an uncertainty band (±30% of expected weekly delta, clipped)
        band = max(0.1, abs(daily_delta_kg * 7.0) * 0.3)
        pts.append(ForecastPoint(
            day=wk * 7,
            date=(start + timedelta(days=wk*7)).isoformat(),
            weight_kg=round(w, 1),
            p10=round(max(0.0, w - band), 1),
            p90=round(w + band, 1),
        ))
    return pts

# Curated citations (keep short, high-quality)
TRUST_SOURCES = [
    Source(
        title="CDC – How much physical activity do adults need?",
        url="https://www.cdc.gov/physicalactivity/basics/adults/index.htm",
    ),
    Source(
        title="Dietary Guidelines for Americans 2020–2025 (sodium ≤2300 mg/day)",
        url="https://www.dietaryguidelines.gov/",
    ),
    Source(
        title="NIH/PMC – Why the 3500-kcal rule is wrong (dynamic energy balance)",
        url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC532583/"),
    Source(
        title="NEJM Editorial (STEP Program: once-weekly semaglutide in obesity)",
        url="https://www.nejm.org/doi/full/10.1056/NEJMe2108274"),
]

####

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

# ---- Routes ----
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/v1/plan", response_model=PlanResponse)
def diet_plan(req: PlanRequest, x_api_key: Optional[str] = Header(default=None)):
    profile = req.dict()
    targets = compute_targets(profile)

    user_ctx = {
        **profile,
        "calorie_target_hint": targets["calorie_target"],
    }

    try:
        out = run_diet(json.dumps(user_ctx, ensure_ascii=False), json.dumps(targets))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"diet agent error: {e}")

    plan_md = out["plan_markdown"]
    why = out.get("why", {})

    # Provenance: you can augment this with real tool logs (FDC IDs, etc.)
    provenance = {
        "tools_used": ["usda_search_foods", "usda_food_details"],  # if present
        "fdc_ids": [],  # (optional) fill in from your tool node outputs if you log them
    }

    # Daily math can be produced by the agent or by you (recommended: you).
    daily_math = {"note": "Per-meal kcal/macros are shown in the table; totals are sums."}

    trust = PlanTrust(
        why_summary = why.get("why_summary",""),
        personalization = why.get("personalization",[]),
        assumptions = why.get("assumptions",[]),
        safety_checks = why.get("safety_checks",[]),
        provenance = provenance,
        daily_math = daily_math,
        forecast = forecast_weight(
            weight_kg=profile["weight_kg"],
            tdee=float(targets["tdee"]),
            calorie_target=float(targets["calorie_target"]),
            weeks=12,
        ),
        citations = TRUST_SOURCES,
    )

    if not plan_md.strip():
        raise HTTPException(status_code=502, detail="No content from diet agent")

    return PlanResponse(targets=targets, plan_markdown=plan_md, trust=trust)


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

##Recipe methods
@app.post("/v1/recipe/suggest", response_model=RecipeSuggestResponse)
def recipe_suggest(req: RecipeSuggestRequest):
    logger.info(f"main.py DEBUG: recipe_suggest")
    try:
        # suggestions = run_recipe_suggest(req.json())
        suggestions = run_recipe_suggest(req.ingredients, req.cuisine, req.count)
        return {"suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"recipe suggest error: {e}")

@app.post("/v1/recipe/detail", response_model=RecipeDetailResponse)
def recipe_detail(req: RecipeDetailRequest):
    logger.info(f"main.py DEBUG: recipe_detail")
    try:
        # detail = run_recipe_detail(req.json())
        detail = run_recipe_detail(req.dish, req.ingredients, req.cuisine)
        return {
            "dish": req.dish,
            "steps": detail.get("steps", []),
            "servings": detail.get("servings", 2),
            "nutrition": detail.get("nutrition", {}),
            "image_url": detail.get("image_url"),
            "image_b64": detail.get("image_b64"),
            "image_error": detail.get("image_error"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"recipe detail error: {e}")
    
# ---- NEW: Coach search schemas ----
class Professional(BaseModel):
    id: Optional[str] = None
    name: str
    role: str
    rating: Optional[float] = None
    reviews: Optional[int] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    link: Optional[str] = None
    distance_km: Optional[float] = None
    snippet: Optional[str] = None
    price_per_session: Optional[float] = None

class CoachSearchRequest(BaseModel):
    zip: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    roles: List[str] = ["Dietitian", "Health Coach"]
    radius_km: int = 20
    max_results: int = 6

class CoachSearchResponse(BaseModel):
    items: List[Professional]

# ---- NEW: route ----
@app.post("/v1/coach/search", response_model=CoachSearchResponse)
def coach_search(req: CoachSearchRequest):
    if not (req.zip or (req.city and req.state)):
        raise HTTPException(status_code=400, detail="Provide either zip or city+state.")

    try:
        items = run_coach_search(
            zip=req.zip,
            city=req.city,
            state=req.state,
            roles=req.roles,
            radius_km=req.radius_km,
            max_results=req.max_results,
        )
        return {"items": items}
    except Exception as e:
        logger.exception("coach_search failed")
        raise HTTPException(status_code=500, detail=f"coach search error: {e}")
    

# --- add Pydantic models (place below existing Plan* models) ---
class ProgressRequest(BaseModel):
    profile: Dict[str, Any]          # same shape you send to /v1/plan
    calorie_target: int              # kcal/day the diet agent suggested
    extra_burn_kcal_per_day: int = 0 # additional exercise burn, avg/day
    weeks: int = 26                  # horizon

class ProgressResponse(BaseModel):
    series: List[Dict[str, Any]]     # weekly points
    assumptions: Dict[str, Any]
    explanation_md: str              # short LLM narrative (optional)

# --- helper: simple dynamic model (Hall-inspired, simplified) ---
def simulate_weight_trajectory(
    profile: Dict[str, Any],
    calorie_target: int,
    extra_burn_kcal_per_day: int,
    weeks: int
) -> Dict[str, Any]:
    """
    Deterministic daily sim with adaptive TDEE:
    - 1 kg ≈ 7700 kcal (mixed tissue; conservative)
    - TDEE adapts downward ~22 kcal/day per kg lost
    """
    start_w = float(profile["weight_kg"])
    h_m = float(profile["height_cm"]) / 100.0

    # Baseline TDEE from your existing helpers
    bmr = mifflin_st_jeor_bmr(profile["sex_assigned_at_birth"], start_w, profile["height_cm"], profile["age"])
    baseline_tdee = tdee_from_activity(bmr, profile["activity_level"])

    kcal_per_kg = 7700.0
    adapt_per_kg = 22.0    # kcal/day per kg lost (typical range ~ 20–30)
    days = max(1, int(weeks) * 7)

    w = start_w
    series: List[Dict[str, Any]] = []
    week_deficit_sum = 0.0

    for day in range(1, days + 1):
        lost_kg = start_w - w
        tdee_now = baseline_tdee - adapt_per_kg * lost_kg
        # Safety floor for TDEE (avoid pathological negatives)
        tdee_now = max(900.0, tdee_now)

        deficit = (tdee_now + float(extra_burn_kcal_per_day)) - float(calorie_target)
        # Positive "deficit" here means energy out > in, so weight goes DOWN
        dw = - deficit / kcal_per_kg
        w = max(30.0, w + dw)  # floor on weight for stability

        week_deficit_sum += deficit

        if day % 7 == 0:
            week = day // 7
            bmi_val = round(w / (h_m ** 2), 1)
            weekly_loss_kg = round((-week_deficit_sum) / kcal_per_kg, 2)  # positive means loss
            series.append({
                "week": week,
                "weight_kg": round(w, 1),
                "bmi": bmi_val,
                "weekly_loss_kg": weekly_loss_kg
            })
            week_deficit_sum = 0.0

    # Milestones
    def first_week_at_or_below(target_w):
        for p in series:
            if p["weight_kg"] <= target_w:
                return p["week"]
        return None

    five_pct_w  = round(start_w * 0.95, 1)
    ten_pct_w   = round(start_w * 0.90, 1)
    five_pct_wk = first_week_at_or_below(five_pct_w)
    ten_pct_wk  = first_week_at_or_below(ten_pct_w)

    assumptions = {
        "start_weight_kg": round(start_w, 1),
        "height_cm": profile["height_cm"],
        "baseline_tdee_kcal": round(baseline_tdee),
        "calorie_target_kcal": int(calorie_target),
        "extra_burn_kcal_per_day": int(extra_burn_kcal_per_day),
        "kcal_per_kg": kcal_per_kg,
        "tdee_adapt_kcal_per_kg": adapt_per_kg,
        "weeks": weeks,
        "milestones": {
            "five_percent_weight_kg": five_pct_w,
            "ten_percent_weight_kg": ten_pct_w,
            "week_to_five_percent": five_pct_wk,
            "week_to_ten_percent": ten_pct_wk,
            "end_weight_kg": series[-1]["weight_kg"] if series else start_w
        }
    }

    return {"series": series, "assumptions": assumptions}

# --- route: projection + LLM explanation ---
from graph.app_graph import run_progress_explainer  # (added below)

@app.post("/v1/progress/project", response_model=ProgressResponse)
def progress_project(req: ProgressRequest):
    try:
        sim = simulate_weight_trajectory(
            profile=req.profile,
            calorie_target=req.calorie_target,
            extra_burn_kcal_per_day=req.extra_burn_kcal_per_day,
            weeks=req.weeks
        )
        expl = run_progress_explainer(
            json.dumps(req.profile, ensure_ascii=False),
            json.dumps(sim["assumptions"], ensure_ascii=False)
        )
        return {
            "series": sim["series"],
            "assumptions": sim["assumptions"],
            "explanation_md": expl or ""
        }
    except Exception as e:
        logger.exception("projection error")
        raise HTTPException(status_code=500, detail=f"projection error: {e}")