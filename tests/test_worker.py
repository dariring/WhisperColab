"""GPU-free unit tests for the native Colab Whisper Large-v3 worker."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from whisper_worker import run_transcription


def mock_audio_preprocessor(source, target, reporter):
    target.write_bytes(b"fake wav, mock model does not decode it")
    reporter("audio", codec="aac", duration=4.0, rms_dbfs=-25.0, message="Audio decoded")
    return {"codec": "aac", "duration": 4.0, "rms_dbfs": -25.0}


class MockModel:
    created_with = None
    calls = []

    def __init__(self, *args, **kwargs):
        MockModel.created_with = (args, kwargs)

    def transcribe(self, filename, **kwargs):
        MockModel.calls.append((filename, kwargs))
        segments = [
            SimpleNamespace(text="안녕하세요.", end=2.0),
            SimpleNamespace(text=" 테스트입니다.", end=4.0),
        ]
        return segments, SimpleNamespace(duration=4.0, language="ko")


def test_whisper_large_v3_transcription(tmp_path):
    src = tmp_path / "korean.m4a"
    src.write_bytes(b"placeholder")
    dst = tmp_path / "korean.txt"
    events = []
    MockModel.calls.clear()
    result = run_transcription(
        src, dst, language="ko", vad=False,
        model_factory=MockModel,
        audio_preprocessor=mock_audio_preprocessor,
        reporter=lambda kind, **kwargs: events.append({"kind": kind, **kwargs}),
    )
    assert dst.read_text(encoding="utf-8") == "안녕하세요. 테스트입니다.\n"
    assert MockModel.created_with == (
        ("large-v3",), {"device": "cuda", "compute_type": "int8_float16"}
    )
    assert MockModel.calls[0][0].endswith(".wav")
    assert MockModel.calls[0][1] == {
        "language": "ko", "beam_size": 5, "vad_filter": False, "task": "transcribe",
    }
    assert result["language"] == "ko"
    assert result["retried_without_vad"] is False
    assert [e["percent"] for e in events if e["kind"] == "progress"] == [50, 99]
    assert events[-1]["kind"] == "done"
    assert events[-1]["percent"] == 100


def test_empty_vad_result_retries_without_filter(tmp_path):
    src = tmp_path / "voice.m4a"
    src.write_bytes(b"x")
    dst = tmp_path / "voice.txt"
    used_vad = []

    class SilenceFilteredModel:
        def __init__(self, *a, **kw):
            pass

        def transcribe(self, filename, **kwargs):
            used_vad.append(kwargs["vad_filter"])
            if kwargs["vad_filter"]:
                return [], SimpleNamespace(duration=0.0, language="ko")
            return [SimpleNamespace(text="음성이 들립니다.", end=4.0)], SimpleNamespace(
                duration=4.0, language="ko"
            )

    events = []
    info = run_transcription(
        src, dst, language="auto", vad=True,
        model_factory=SilenceFilteredModel,
        audio_preprocessor=mock_audio_preprocessor,
        reporter=lambda kind, **kw: events.append((kind, kw)),
    )
    assert used_vad == [True, False]
    assert info["retried_without_vad"] is True
    assert dst.read_text(encoding="utf-8") == "음성이 들립니다.\n"
    assert any(k == "warning" for k, _ in events)


@pytest.mark.parametrize("vad,expected_calls", [(True, 2), (False, 1)])
def test_still_empty_is_error_without_txt(tmp_path, vad, expected_calls):
    src = tmp_path / "silence.m4a"
    src.write_bytes(b"x")
    dst = tmp_path / "silence.txt"
    dst.write_text("previous run", encoding="utf-8")
    calls = []

    class NoSpeechModel:
        def __init__(self, *a, **kw):
            pass

        def transcribe(self, filename, **kwargs):
            calls.append(kwargs["vad_filter"])
            return [SimpleNamespace(text="   ", end=4.0)], SimpleNamespace(
                duration=4.0, language="ko"
            )

    with pytest.raises(RuntimeError, match="음성 인식 결과가 비어 있습니다"):
        run_transcription(
            src, dst, vad=vad,
            model_factory=NoSpeechModel,
            audio_preprocessor=mock_audio_preprocessor,
            reporter=lambda *a, **kw: None,
        )
    assert len(calls) == expected_calls
    assert not dst.exists()


def test_missing_file_and_invalid_language(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_transcription(tmp_path / "missing.m4a", tmp_path / "output.txt")
    src = tmp_path / "audio.m4a"
    src.touch()
    with pytest.raises(ValueError, match="Unsupported recognition language"):
        run_transcription(src, tmp_path / "output.txt", language="xx")


def test_notebook_uses_native_colab_upload():
    notebook = json.loads(Path("WhisperColab.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "files.upload()" in code
    assert "files.download(" in code
    assert "subprocess.Popen(" in code
    assert "whisper_worker.py" in code
    assert "widgets.FileUpload" not in code
    assert "Colab 파일 경로 사용" in code
    assert "자동_TXT_다운로드" in code
    assert "cloudflared" not in code
    assert "uvicorn" not in code
    assert "trycloudflare" not in code.lower()
    assert "not full_text.strip()" in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "WhisperColab.ipynb", "exec")
