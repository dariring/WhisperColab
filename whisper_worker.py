"""Whisper Large-v3 CLI worker for Google Colab's *notebook* interface.

No web server, tunnel, socket listener or OpenAI API is used.
Each JSON line on stdout is a progress event consumed by the notebook.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Any


def emit(kind: str, **data: Any) -> None:
    print(json.dumps({"kind": kind, **data}, ensure_ascii=False), flush=True)


def run_transcription(
    input_file: Path,
    output_file: Path,
    *,
    language: str = "auto",
    vad: bool = True,
    model_factory: Callable[..., Any] | None = None,
    reporter: Callable[..., None] = emit,
) -> dict[str, Any]:
    """Transcribe one audio file; keep side effects confined to its output path."""
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if language not in {"auto", "ko", "en", "ja", "zh"}:
        raise ValueError("Unsupported recognition language")
    if model_factory is None:
        from faster_whisper import WhisperModel
        model_factory = WhisperModel

    reporter("status", message="Whisper Large-v3 모델 불러오는 중 (최초 실행 시 다운로드)")
    model = model_factory("large-v3", device="cuda", compute_type="int8_float16")
    reporter("status", message="음성 인식 시작")

    segments, info = model.transcribe(
        str(input_file),
        language=None if language == "auto" else language,
        beam_size=5,
        vad_filter=vad,
    )
    duration = max(float(info.duration or 0), 0.01)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_file.open("w", encoding="utf-8") as out:
        for segment in segments:
            out.write(segment.text)
            written += len(segment.text)
            progress = min(99, max(1, int(float(segment.end) / duration * 100)))
            reporter("progress", percent=progress, message=f"음성 인식 중 · {progress}%")
        if written:
            out.write("\n")

    reporter("done", percent=100, language=info.language, characters=written)
    return {"language": info.language, "characters": written, "output": str(output_file)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper Large-v3 on GPU")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", choices=("auto", "ko", "en", "ja", "zh"), default="auto")
    parser.add_argument("--no-vad", action="store_true", help="Disable silence filtering")
    args = parser.parse_args()

    try:
        run_transcription(
            args.input, args.output, language=args.language, vad=not args.no_vad
        )
    except Exception as exc:
        # Fail with a nonzero exit status and an event the notebook can show.
        emit("error", message=f"{type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
