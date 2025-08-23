# backend/tools/coach_tools.py
import os, json, requests
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool

SERPAPI_KEY = os.getenv("SERPAPI_KEY")

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _serp_maps_search(q: str, max_results: int) -> List[Dict[str, Any]]:
    """
    Use SerpAPI Google Maps engine to search. Returns simplified list.
    """
    logger.info("DEBUG: SERP API SEARCH")

    if not SERPAPI_KEY:
        return [{"error": "SERPAPI_KEY not set"}]

    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_maps",
        "q": q,
        "api_key": SERPAPI_KEY,
        # You can add "type":"search" and other params if needed.
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    # The schema can be 'local_results' or 'place_results' depending on query.
    items = data.get("local_results") or []
    out: List[Dict[str, Any]] = []
    for it in items[:max_results]:
        out.append({
            "id": it.get("place_id") or it.get("position"),
            "name": it.get("title"),
            "role": ", ".join(it.get("type", [])) if isinstance(it.get("type"), list) else (it.get("type") or ""),
            "rating": it.get("rating"),
            "reviews": it.get("reviews"),
            "address": it.get("address"),
            "phone": it.get("phone"),
            "website": it.get("website"),
            "link": it.get("link"),
            # distance is not always available; leaving as None
        })
    return out

@tool("search_local_pros", return_direct=False)
def search_local_pros(
    zip: Optional[str] = None,
    city: Optional[str] = None,
    state: Optional[str] = None,
    roles: List[str] = ("Dietitian", "Health Coach"),
    radius_km: int = 20,  # reserved for future filtering when we compute distance
    max_results: int = 6,
) -> str:
    """
    Search for local professionals using SerpAPI Google Maps.
    Provide either zip or city+state. Roles controls query terms (e.g., Dietitian, Health Coach, Nutritionist).
    Returns JSON: {"items":[...]} where each item: name, role, rating, reviews, address, phone, website, link.
    """
    logger.info("DEBUG: search_local_pros")
    # Build a location string for the query
    loc = zip or " ".join([c for c in [city, state] if c]) or ""
    if not loc:
        return json.dumps({"error": "missing location"}, ensure_ascii=False)

    # Merge role queries
    results: List[Dict[str, Any]] = []
    seen: set[str] = set()

    role_terms = roles or ["Dietitian", "Health Coach"]
    for term in role_terms:
        q = f"{term} near {loc}"
        try:
            found = _serp_maps_search(q, max_results)
        except Exception as e:
            found = [{"error": f"serpapi error: {e}"}]

        for f in found:
            # Deduplicate by name+address if both present
            key = f"{f.get('name','').strip().lower()}|{f.get('address','').strip().lower()}"
            if key in seen:
                continue
            seen.add(key)
            results.append(f)

        if len(results) >= max_results:
            break

    # Trim to max_results
    results = results[:max_results]
    return json.dumps({"items": results}, ensure_ascii=False)
