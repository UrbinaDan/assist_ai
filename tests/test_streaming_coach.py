from app.ws import _chunk_stream_text, _coach_stream_text, _should_stream_coach


def test_coach_stream_text_uses_final_payload_without_changing_contract():
    out = {
        "kind": "final",
        "response_type": "coach_final",
        "coach_final": {
            "suggestions": ["Use a concrete project example.", "Tie the result to the role."],
            "follow_up": "Can you quantify the result?",
            "bridge": "Connect the project to backend reliability.",
        },
    }

    text = _coach_stream_text(out)

    assert "1. Use a concrete project example." in text
    assert "Follow-up: Can you quantify the result?" in text
    assert "Bridge: Connect the project to backend reliability." in text


def test_coach_stream_chunks_have_delta_text():
    chunks = list(_chunk_stream_text("abcdef", chunk_size=2))
    assert chunks == ["ab", "cd", "ef"]


def test_streaming_disabled_by_default_for_final_payload():
    out = {
        "kind": "final",
        "response_type": "coach_final",
        "coach_final": {"suggestions": ["Answer"]},
    }
    assert _should_stream_coach(out) is False
