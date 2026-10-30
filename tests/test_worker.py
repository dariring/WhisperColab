"""GPU-free tests for the Colab-native Whisper Large-v3 worker."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from whisper_worker import run_transcription


class MockModel:
    created_with = None
    transcribed_with = None

    def __init__(self, *args, **kwargs):
        MockModel.created_with = (args, kwargs)

    def transcribe(self, filename, **kwargs):
        MockModel.transcribed_with = (filename, kwargs)
        segments = [
            SimpleNamespace(text="안녕하세요.", end=2.0),
            SimpleNamespace(text=" 테스트입니다.", end=4.0),
        ]
        return segments, SimpleNamespace(duration=4.0, language="ko")


def test_whisper_large_v3_transcription(tmp_path):
    source = tmp_path / "korean.mp3"
    source.write_bytes(b"mock audio contents")
    destination = tmp_path / "results" / "korean.txt"
    events = []
    result = run_transcription(
        source, destination, language="ko", vad=False,
        model_factory=MockModel,
        reporter=lambda kind, **kwargs: events.append({"kind": kind, **kwargs}),
    )
    assert destination.read_text(encoding="utf-8") == "안녕하세요. 테스트입니다.\n"
    assert MockModel.created_with == (
        ("large-v3",), {"device": "cuda", "compute_type": "int8_float16"}
    )
    assert MockModel.transcribed_with[1] == {
        "language": "ko", "beam_size": 5, "vad_filter": False,
    }
    assert result["language"] == "ko"
    assert events[0]["kind"] == "status"
    assert [event["percent"] for event in events if event["kind"] == "progress"] == [50, 99]
    assert events[-1]["kind"] == "done"
    assert events[-1]["percent"] == 100


def test_auto_detect_and_empty_result(tmp_path):
    source = tmp_path / "silence.wav"
    source.touch()
    target = tmp_path / "silence.txt"

    class SilentModel:
        def __init__(self, *a, **kw):
            pass

        def transcribe(self, filename, **kwargs):
            assert kwargs["language"] is None
            assert kwargs["vad_filter"] is True
            return [], SimpleNamespace(duration=0, language="en")

    result = run_transcription(
        source, target, model_factory=SilentModel, reporter=lambda *a, **kw: None
    )
    assert target.read_text(encoding="utf-8") == ""
    assert result["characters"] == 0


def test_missing_file_and_invalid_language(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_transcription(tmp_path / "missing.mp3", tmp_path / "output.txt")
    src = tmp_path / "audio.mp3"
    src.touch()
    with pytest.raises(ValueError, match="Unsupported recognition language"):
        run_transcription(src, tmp_path / "output.txt", language="xx")


def test_notebook_uses_native_widgets_not_public_tunnel():
    notebook = json.loads(Path("WhisperColab.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "widgets.FileUpload" in code
    assert "uploader.observe" in code
    assert "files.download(" in code
    assert "threading.Thread" in code
    assert "whisper_worker.py" in code
    assert "cloudflared" not in code
    assert "uvicorn" not in code
    assert "trycloudflare" not in code.lower()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "WhisperColab.ipynb", "exec")
