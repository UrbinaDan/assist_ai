"""DeltaIn accepts optional WebSocket fields for audio/speaker mode (Phase X)."""
from app.schemas import DeltaIn


def test_delta_in_audio_fields_optional():
    d = DeltaIn(session_id="s1")
    assert d.audio_source is None
    assert d.speaker_mode is None
    assert d.client_sent_at is None


def test_delta_in_accepts_audio_fields():
    d = DeltaIn(
        session_id="s2",
        audio_source="mic",
        speaker_mode="source",
        speaker="Me",
        final=True,
    )
    assert d.audio_source == "mic"
    assert d.speaker_mode == "source"


def test_delta_in_accepts_client_sent_at():
    d = DeltaIn(session_id="s3", client_sent_at=123.456)
    assert d.client_sent_at == 123.456
