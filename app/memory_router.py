from __future__ import annotations

from typing import Any, Dict, List


DEFAULT_MEMORY_TYPES = ["semantic", "episodic", "procedural", "preferences"]

ROUTE_MEMORY_TYPES = {
    "behavioral_story": ["episodic", "procedural", "preferences"],
    "technical_project": ["semantic", "episodic", "procedural", "preferences"],
    "personal_background": ["semantic", "episodic", "preferences"],
    "company_fit": ["semantic", "episodic", "preferences"],
    "coding_interview": ["procedural", "semantic"],
    "system_design": ["procedural", "semantic"],
    "direct_knowledge": ["semantic", "procedural"],
    "small_talk": ["semantic", "preferences"],
    "notes": [],
}


def memory_filters_for_route(route: str) -> Dict[str, Any]:
    normalized = (route or "unknown").strip()
    memory_types: List[str] = ROUTE_MEMORY_TYPES.get(normalized, DEFAULT_MEMORY_TYPES)
    return {
        "memory_types": list(memory_types),
        "topics": [],
        "min_priority": 0.0,
    }
