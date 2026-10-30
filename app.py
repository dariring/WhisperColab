"""Whisper Large-v3 transcription panel for a temporary Colab session.

HTTP polling is used deliberately: Cloudflare Quick Tunnels do not support SSE.
"""
from __future__ import annotations

import hmac
import logging
import os
import re
import tempfile
import threading
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = Path(tempfile.mkdtemp(prefix="whispercolab-"))
MAX_UPLOAD_BYTES = 75 * 1024 * 1024
JOB_TTL_SECONDS = 60 * 60
MAX_ACTIVE_JOBS = 3
SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus",
    ".aac", ".mp4", ".webm", ".wma", ".aiff", ".mkv",
}
LANGUAGES = {"auto": None, "ko": "ko", "en": "en", "ja": "ja", "zh": "zh"}

app = FastAPI(title="WhisperColab", docs_url=None, redoc_url=None, openapi_url=None)
basic_auth = HTTPBasic(auto_error=False)
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
jobs_lock = threading.Lock()
jobs: dict[str, dict] = {}
_model = None


def require_auth(credentials: HTTPBasicCredentials | None = Depends(basic_auth)) -> None:
    password = os.environ.get("PANEL_PASSWORD", "")
    if not password:
        raise HTTPException(status_code=503, detail="PANEL_PASSWORD is not configured")
    valid = (
        credentials is not None
        and hmac.compare_digest(credentials.username.encode("utf-8"), b"whisper")
        and hmac.compare_digest(credentials.password.encode("utf-8"), password.encode("utf-8"))
    )
    if not valid:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": 'Basic realm="WhisperColab"'},
        )


def safe_stem(filename: str) -> str:
    name = os.path.basename(filename.replace("\\", "/"))
    stem = unicodedata.normalize("NFKC", Path(name).stem)
    stem = re.sub(r"[^\w .()-]", "_", stem, flags=re.UNICODE).strip(" ._")
    return (stem[:90] or "transcript")


def cleanup_jobs() -> None:
    now = time.monotonic()
    for job_id, job in list(jobs.items()):
        if job["state"] in {"done", "error"} and now - job["updated"] > JOB_TTL_SECONDS:
            Path(job["audio_path"]).unlink(missing_ok=True)
            Path(job["txt_path"]).unlink(missing_ok=True)
            del jobs[job_id]


def update_job(job_id: str, **fields) -> None:
    with jobs_lock:
        job = jobs.get(job_id)
        if job:
            job.update(fields)
            job["updated"] = time.monotonic()


def transcribe_job(job_id: str) -> None:
    global _model
    with jobs_lock:
        job = jobs[job_id]
        input_path = job["audio_path"]
        requested_language = job["requested_language"]
        use_vad = job["vad"]
        txt_path = job["txt_path"]
    try:
        if _model is None:
            update_job(job_id, state="loading", message="Whisper Large-v3 모델 준비 중 (최초 실행 시 다운로드)")
            from faster_whisper import WhisperModel
            _model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")

        update_job(job_id, state="running", progress=1, message="음성을 분석하고 있습니다")
        segments, info = _model.transcribe(
            input_path,
            language=LANGUAGES[requested_language],
            beam_size=5,
            vad_filter=use_vad,
        )
        total_duration = max(float(info.duration), 0.01)
        parts: list[str] = []
        for segment in segments:
            parts.append(segment.text)
            pct = max(1, min(99, int(segment.end / total_duration * 100)))
            update_job(
                job_id,
                progress=pct,
                message="음성 인식 중 · " + str(pct) + "%",
            )

        transcript = "".join(parts).strip()
        Path(txt_path).write_text(transcript + ("\n" if transcript else ""), encoding="utf-8")
        update_job(
            job_id,
            state="done",
            progress=100,
            message="변환이 완료되었습니다",
            language=info.language,
            preview=transcript[:100_000],
            preview_truncated=len(transcript) > 100_000,
        )
    except Exception:
        logging.exception("Transcription failed for job %s", job_id)
        update_job(
            job_id,
            state="error",
            message="변환에 실패했습니다. 오디오 형식, GPU 상태, Colab 로그를 확인해 주세요.",
        )
    finally:
        Path(input_path).unlink(missing_ok=True)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/", response_class=HTMLResponse, dependencies=[Depends(require_auth)])
def homepage():
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.post("/api/jobs", status_code=202, dependencies=[Depends(require_auth)])
async def submit(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    vad: bool = Form(True),
):
    if language not in LANGUAGES:
        raise HTTPException(status_code=400, detail="지원하지 않는 언어입니다")
    filename = file.filename or ""
    extension = Path(filename.lower()).suffix
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다")

    job_id = uuid.uuid4().hex
    input_path = WORK_DIR / (job_id + extension)
    txt_path = WORK_DIR / (job_id + ".txt")
    with jobs_lock:
        cleanup_jobs()
        active = sum(j["state"] not in {"done", "error"} for j in jobs.values())
        if active >= MAX_ACTIVE_JOBS:
            raise HTTPException(status_code=429, detail="대기열이 가득 찼습니다. 잠시 후 다시 시도해 주세요.")
        jobs[job_id] = {
            "state": "uploading",
            "progress": 0,
            "message": "업로드 중",
            "updated": time.monotonic(),
            "audio_path": str(input_path),
            "txt_path": str(txt_path),
            "filename": safe_stem(filename),
            "requested_language": language,
            "vad": vad,
            "language": None,
            "preview": "",
            "preview_truncated": False,
        }

    try:
        size = 0
        with input_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="파일 크기는 75MB 이하만 지원합니다")
                destination.write(chunk)
        if not size:
            raise HTTPException(status_code=400, detail="빈 파일은 업로드할 수 없습니다")
    except Exception:
        input_path.unlink(missing_ok=True)
        with jobs_lock:
            jobs.pop(job_id, None)
        raise
    finally:
        await file.close()

    update_job(job_id, state="queued", message="변환 대기 중")
    executor.submit(transcribe_job, job_id)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}", dependencies=[Depends(require_auth)])
def job_status(job_id: str):
    with jobs_lock:
        cleanup_jobs()
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="작업을 찾을 수 없거나 보관 기간이 지났습니다")
        return {
            "id": job_id,
            "state": job["state"],
            "progress": job["progress"],
            "message": job["message"],
            "language": job["language"],
            "text": job["preview"] if job["state"] == "done" else "",
            "preview_truncated": job["preview_truncated"],
            "download_url": "/api/jobs/" + job_id + "/txt" if job["state"] == "done" else None,
        }


@app.get("/api/jobs/{job_id}/txt", dependencies=[Depends(require_auth)])
def download_txt(job_id: str):
    with jobs_lock:
        cleanup_jobs()
        job = jobs.get(job_id)
        if not job or job["state"] != "done":
            raise HTTPException(status_code=404, detail="다운로드 파일이 없습니다")
        path = job["txt_path"]
        filename = job["filename"] + ".txt"
    return FileResponse(path, media_type="text/plain; charset=utf-8", filename=filename)
