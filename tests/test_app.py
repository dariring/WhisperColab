"""API contract tests; no Colab GPU or model download required."""
import time
from types import SimpleNamespace

import app as service
from fastapi.testclient import TestClient

client = TestClient(service.app)
AUTH = ("whisper", "testing-password")


class FakeModel:
    def transcribe(self, path, *, language, beam_size, vad_filter):
        assert path.endswith(".mp3")
        assert beam_size == 5
        return (
            [SimpleNamespace(text="안녕하세요.", end=1.0),
             SimpleNamespace(text=" 테스트입니다.", end=2.0)],
            SimpleNamespace(duration=2.0, language="ko"),
        )


def test_requires_authentication(monkeypatch):
    monkeypatch.setenv("PANEL_PASSWORD", "testing-password")
    assert client.get("/").status_code == 401
    assert client.get("/", auth=("whisper", "wrong")).status_code == 401
    assert client.get("/", auth=AUTH).status_code == 200
    assert client.get("/api/jobs/nonexistent", auth=AUTH).status_code == 404


def test_upload_transcribe_and_download(monkeypatch):
    monkeypatch.setenv("PANEL_PASSWORD", "testing-password")
    monkeypatch.setattr(service, "_model", FakeModel())
    response = client.post(
        "/api/jobs",
        auth=AUTH,
        data={"language": "ko", "vad": "true"},
        files={"file": ("회의녹음.mp3", b"pretend-audio", "audio/mpeg")},
    )
    assert response.status_code == 202
    job_id = response.json()["id"]

    result = None
    for _ in range(100):
        result = client.get("/api/jobs/" + job_id, auth=AUTH).json()
        if result["state"] in ("done", "error"):
            break
        time.sleep(0.02)
    assert result["state"] == "done", result
    assert result["progress"] == 100
    assert result["text"] == "안녕하세요. 테스트입니다."
    assert result["language"] == "ko"

    download = client.get(result["download_url"], auth=AUTH)
    assert download.status_code == 200
    assert download.content.decode("utf-8") == "안녕하세요. 테스트입니다.\n"
    assert "attachment" in download.headers["content-disposition"]


def test_rejects_invalid_uploads(monkeypatch):
    monkeypatch.setenv("PANEL_PASSWORD", "testing-password")
    invalid_type = client.post(
        "/api/jobs",
        auth=AUTH,
        files={"file": ("bad.exe", b"xxx", "application/octet-stream")},
    )
    assert invalid_type.status_code == 400

    empty = client.post(
        "/api/jobs",
        auth=AUTH,
        files={"file": ("empty.mp3", b"", "audio/mpeg")},
    )
    assert empty.status_code == 400

    invalid_language = client.post(
        "/api/jobs",
        auth=AUTH,
        data={"language": "unknown"},
        files={"file": ("ok.mp3", b"xxx", "audio/mpeg")},
    )
    assert invalid_language.status_code == 400


def test_filename_sanitization():
    assert service.safe_stem(r"..\folder\회의녹음.mp3") == "회의녹음"
    assert service.safe_stem("a<>b.wav") == "a__b"
