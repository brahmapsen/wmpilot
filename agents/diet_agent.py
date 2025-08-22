# backend/agents/diet_agent.py
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from tools.nutrition_tools import (
    usda_search_foods,
    usda_food_details,
    nutritionix_parse,
)

load_dotenv()
AIML_API_KEY = os.getenv("AIML_API_KEY", "")
AIML_BASE_URL = os.getenv("AIML_BASE_URL", "https://api.aimlapi.com/v1")
MODEL = os.getenv("DIET_MODEL", "openai/gpt-4o")  # 4o works well with tools via AIML

SYSTEM_PROMPT = """You are a nutrition & weight-management assistant for educational use.
- Use Mifflin-St Jeor for BMR, an activity factor for TDEE (provided by server).
- Goals: maintenance (±0), loss (-10% to -20%), gain (+10% to +20%).
- Default macros: 25–35% protein, 30–40% carbs, 25–35% fat.
- Respect allergies, dietary restrictions, and cultural preferences.
- Prefer whole foods, moderate added sugars, sodium ~≤2300 mg/day unless otherwise specified.
- When grounding nutrients, call USDA tools sparingly (representative items).
- Return: a short intro + 7-day Markdown table (breakfast/lunch/dinner/snack) with kcal/macros per meal & daily totals. End with a brief safety disclaimer.
"""

# ---- Graph State ----
from typing import TypedDict
from langchain_core.messages import BaseMessage

class DietState(TypedDict):
    messages: List[BaseMessage]

# ---- LLM with tools ----
def diet_llm():
    return ChatOpenAI(
        model=MODEL,
        api_key=AIML_API_KEY,
        base_url=AIML_BASE_URL,
        temperature=0.2,
        max_tokens=1200,
    ).bind_tools([usda_search_foods, usda_food_details, nutritionix_parse])

tool_node = ToolNode([usda_search_foods, usda_food_details, nutritionix_parse])

def llm_node(state: DietState):
    llm = diet_llm()
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def build_diet_graph():
    graph = StateGraph(DietState)
    graph.add_node("diet_llm", llm_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("diet_llm")
    # If LLM asked for tools -> ToolNode -> back to diet_llm; else end
    graph.add_conditional_edges("diet_llm", tools_condition, {  # True/False mapping handled internally
        "tools": "tools",
        END: END,
    })
    graph.add_edge("tools", "diet_llm")
    return graph.compile()
