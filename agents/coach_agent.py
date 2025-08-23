# backend/agents/coach_agent.py  (patch)

import os
import logging
from typing import List, Sequence, TypedDict, Annotated, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage, HumanMessage, BaseMessage, AIMessage, ToolMessage
)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

from tools.coach_tools import search_local_pros

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
AIML_API_KEY = os.getenv("AIML_API_KEY", "")
AIML_BASE_URL = os.getenv("AIML_BASE_URL", "https://api.aimlapi.com/v1")
MODEL = os.getenv("COACH_MODEL", "openai/gpt-4o")

SYSTEM_PROMPT = """You are a locator assistant for dietitians, nutritionists, and health coaches.
- You have a tool `search_local_pros` that performs a Google Maps (SerpAPI) query.
- Call the tool ONCE per request with the provided location and roles.
- Return a compact JSON array of up to max_results items where each item has:
  name, role, rating (float?), reviews (int?), address, phone, website, link, distance_km (if you have it).
- If the tool returns more fields, keep only the useful ones.
- Do NOT return markdown—return pure JSON only.
"""

class CoachState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def coach_llm():
    logger.info("coach agent: LLM call (tools enabled)")
    return ChatOpenAI(
        model=MODEL,
        api_key=AIML_API_KEY,
        base_url=AIML_BASE_URL,
        temperature=0.1,
        max_tokens=900,
    ).bind_tools([search_local_pros])

# NEW: for the second turn, no tools bound (prevents tool_calls => content:null)
def final_llm():
    logger.info("coach agent: LLM call (finalizer; tools disabled)")
    return ChatOpenAI(
        model=MODEL,
        api_key=AIML_API_KEY,
        base_url=AIML_BASE_URL,
        temperature=0.1,
        max_tokens=900,
    )

tool_node = ToolNode([search_local_pros])

def _sanitize_for_openai(msgs: List[BaseMessage]) -> List[BaseMessage]:
    if not msgs:
        return msgs
    i = 0
    while i < len(msgs) and isinstance(msgs[i], ToolMessage):
        i += 1
    msgs = msgs[i:]
    if not msgs:
        return []
    seen_ai_with_tools = False
    cleaned: List[BaseMessage] = []
    for m in msgs:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            seen_ai_with_tools = True
            cleaned.append(m)
        elif isinstance(m, ToolMessage):
            if seen_ai_with_tools:
                cleaned.append(m)
            else:
                continue
        else:
            cleaned.append(m)
    return cleaned or msgs

def _normalize_message_contents(msgs: List[BaseMessage]) -> List[BaseMessage]:
    norm: List[BaseMessage] = []
    for m in msgs:
        if isinstance(m, AIMessage):
            content = m.content if (m.content not in (None, [])) else ""
            norm.append(AIMessage(
                content=content,
                additional_kwargs=getattr(m, "additional_kwargs", {}),
                tool_calls=getattr(m, "tool_calls", None),
            ))
        elif isinstance(m, ToolMessage):
            content = m.content if m.content is not None else ""
            norm.append(ToolMessage(
                content=str(content),
                tool_call_id=m.tool_call_id,
                name=getattr(m, "name", None),
            ))
        else:
            if getattr(m, "content", None) is None:
                if isinstance(m, SystemMessage):
                    norm.append(SystemMessage(content=""))
                elif isinstance(m, HumanMessage):
                    norm.append(HumanMessage(content=""))
                else:
                    norm.append(m)
            else:
                norm.append(m)
    return norm

# NEW: collapse (assistant tool_calls + tool message) into a plain text message
def _collapse_tool_turn(msgs: List[BaseMessage]) -> Optional[List[BaseMessage]]:
    """
    If the latest messages are: ... AIMessage(tool_calls=...), ToolMessage(...),
    replace that pair with a single HumanMessage carrying tool output as text.
    Return a new list when collapse happened, else None.
    """
    if len(msgs) < 2:
        return None
    if not isinstance(msgs[-1], ToolMessage):
        return None
    if not isinstance(msgs[-2], AIMessage) or not getattr(msgs[-2], "tool_calls", None):
        return None

    tool_msg: ToolMessage = msgs[-1]
    ai_tool_call: AIMessage = msgs[-2]
    tool_name = None
    try:
        # best-effort to name the tool
        tcs = getattr(ai_tool_call, "tool_calls", []) or []
        if tcs and isinstance(tcs[0], dict):
            tool_name = tcs[0].get("function", {}).get("name") or tcs[0].get("name")
        elif tcs and hasattr(tcs[0], "name"):
            tool_name = tcs[0].name
    except Exception:
        pass
    tool_name = tool_name or "search_local_pros"

    tool_text = str(tool_msg.content or "").strip()
    # Build a compact instruction so the model finishes without calling tools again
    summary = (
        f"Tool `{tool_name}` returned the following JSON:\n"
        f"{tool_text}\n\n"
        "Return a FINAL, compact JSON array (no markdown) with the best up to max_results items. "
        "Do NOT call any tools. Keep fields: name, role, rating, reviews, address, phone, website, link, distance_km (if available)."
    )

    collapsed = list(msgs[:-2])  # keep everything before the tool turn
    collapsed.append(HumanMessage(content=summary))
    return collapsed

def llm_node(state: CoachState):
    raw = list(state.get("messages", []))
    logger.info("coach_llm: received %d message(s)", len(raw))
    logger.info("coach_llm: roles -> %s", [m.__class__.__name__ for m in raw])

    # First pass: normal sanitize
    safe_msgs = _sanitize_for_openai(raw)

    # If the latest turn is tool_calls -> tool, collapse to plain text & finalize with no tools
    collapsed = _collapse_tool_turn(safe_msgs)
    if collapsed is not None:
        safe_msgs2 = _normalize_message_contents(collapsed)
        if not safe_msgs2:
            safe_msgs2 = [SystemMessage(content=SYSTEM_PROMPT)]
        llm = final_llm()  # <-- no tools bound here
        resp = llm.invoke(safe_msgs2)
        return {"messages": [resp]}

    # Otherwise this is the first turn: tools are allowed
    safe_msgs = _normalize_message_contents(safe_msgs)
    if not safe_msgs:
        safe_msgs = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content="Find local professionals. Use the tool once and return pure JSON.")
        ]
    llm = coach_llm()  # tools bound
    resp = llm.invoke(safe_msgs)
    return {"messages": [resp]}

def build_coach_graph():
    g = StateGraph(CoachState)
    g.add_node("coach_llm", llm_node)
    g.add_node("tools", tool_node)
    g.set_entry_point("coach_llm")
    g.add_conditional_edges("coach_llm", tools_condition, {
        "tools": "tools",
        END: END,
    })
    g.add_edge("tools", "coach_llm")
    return g.compile()
