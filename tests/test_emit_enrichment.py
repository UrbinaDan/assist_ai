"""Optional emit fields for router / interpreter prep (Phase H)."""
from app.agent import _emit_payload, AgentState


def test_emit_payload_includes_route_retrieval_latency():
    st = AgentState(session_id="e1")
    st.retrieval_cache = {"doc_ids": ["chunk::1", "chunk::2"]}
    out = _emit_payload(
        kind="final",
        response_type="coach_final",
        speaker="Interviewer",
        transcript="Hello",
        usage={"turn": {}},
        route_info={"route": "behavioral_story", "confidence": 0.82, "rationale": ""},
        state=st,
        interpreted_question=None,
    )
    assert out["route"] == "behavioral_story"
    assert out["route_confidence"] == 0.82
    assert out["retrieval_source"] == ["chunk::1", "chunk::2"]
    assert isinstance(out.get("latency"), dict)
    assert out["latency"].get("backend_sent_at") is not None
    assert "backend_total_ms" in out["latency"]
    assert "interpreter_ms" in out["latency"]
