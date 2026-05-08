from app.memory_router import memory_filters_for_route
from app.retriever import _matches_filters


def test_behavioral_story_memory_filters():
    out = memory_filters_for_route("behavioral_story")
    assert out["memory_types"] == ["episodic", "procedural", "preferences"]
    assert out["topics"] == []
    assert out["min_priority"] == 0.0


def test_technical_project_memory_filters():
    out = memory_filters_for_route("technical_project")
    assert out["memory_types"] == ["semantic", "episodic", "procedural", "preferences"]


def test_coding_interview_memory_filters():
    out = memory_filters_for_route("coding_interview")
    assert out["memory_types"] == ["procedural", "semantic"]


def test_unknown_memory_filters_default_all_types():
    out = memory_filters_for_route("unknown")
    assert out["memory_types"] == ["semantic", "episodic", "procedural", "preferences"]


def test_notes_memory_filters_empty():
    out = memory_filters_for_route("notes")
    assert out["memory_types"] == []


def test_metadata_filter_matches_memory_type_topic_and_priority():
    meta = {
        "memory_type": "episodic",
        "topic": "debugging_stories",
        "priority": 0.9,
    }
    filters = {
        "memory_types": ["episodic"],
        "topics": ["debugging_stories"],
        "min_priority": 0.8,
    }
    assert _matches_filters(meta, filters) is True


def test_metadata_filter_rejects_wrong_memory_type():
    meta = {
        "memory_type": "semantic",
        "topic": "debugging_stories",
        "priority": 0.9,
    }
    filters = {
        "memory_types": ["episodic"],
        "topics": [],
        "min_priority": 0.0,
    }
    assert _matches_filters(meta, filters) is False
