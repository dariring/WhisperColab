"""Regression coverage for faster-whisper 1.2.1 + PyAV metadata_errors API."""
from io import BytesIO
from pathlib import Path
import wave

import av


def test_pinned_pyav_can_decode_wave_with_metadata_errors():
    """Match the av.open call made by faster-whisper's audio decoder."""
    audio = BytesIO()
    with wave.open(audio, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)

    with av.open(BytesIO(audio.getvalue()), mode="r", metadata_errors="ignore") as container:
        assert len(container.streams.audio) == 1
        decoded_frames = list(container.decode(audio=0))
        assert decoded_frames


def test_runtime_and_test_dependencies_pin_compatible_pyav():
    runtime = Path("requirements.txt").read_text(encoding="utf-8")
    development = Path("requirements-dev.txt").read_text(encoding="utf-8")
    assert "av==17.1.0" in runtime
    assert "av==17.1.0" in development
    assert av.__version__ == "17.1.0"
