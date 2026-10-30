"""Notebook-only Whisper Large-v3 worker with audio checks and safe blank-result handling.

M4A/AAC (and other compressed formats) are first decoded by ffmpeg to a known
16-kHz mono PCM WAV. Progress/status are newline-delimited JSON on stdout.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Callable


def emit(kind: str, **data: Any) -> None:
    print(json.dumps({"kind": kind, **data}, ensure_ascii=False), flush=True)


def prepare_audio(
    input_file: Path,
    decoded_wav: Path,
    *,
    reporter: Callable[..., None] = emit,
) -> dict[str, Any]:
    """Extract the first real audio stream and verify decoded PCM samples."""
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels",
            "-show_entries", "format=duration", "-of", "json",
            str(input_file),
        ],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError("파일 정보를 읽을 수 없습니다: " + probe.stderr.strip()[-400:])
    try:
        metadata = json.loads(probe.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("오디오 정보를 해석할 수 없습니다") from exc
    streams = metadata.get("streams", [])
    if not streams:
        raise ValueError("파일에 오디오 트랙이 없습니다")
    codec = streams[0].get("codec_name", "?")
    reporter("status", message=f"오디오 트랙 확인: {codec} · 16kHz WAV로 변환 중")

    transcode = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
            "-y", "-i", str(input_file), "-map", "0:a:0", "-vn",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(decoded_wav),
        ],
        capture_output=True, text=True, check=False,
    )
    if transcode.returncode != 0:
        raise RuntimeError("오디오 디코딩 실패: " + transcode.stderr.strip()[-500:])

    # Stream the WAV in chunks instead of loading a potentially large M4A into RAM.
    with wave.open(str(decoded_wav), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 16000:
            raise ValueError("오디오 변환 후 파일 형식이 올바르지 않습니다")
        duration = wav.getnframes() / 16000.0
        if duration < 0.1:
            raise ValueError("파일에 분석 가능한 길이의 오디오가 없습니다")
        peak = 0
        total_squared = 0.0
        count = 0
        from array import array
        while block := wav.readframes(16000 * 10):
            samples = array("h")
            samples.frombytes(block)
            if sys.byteorder != "little":
                samples.byteswap()
            count += len(samples)
            peak = max(peak, max((abs(s) for s in samples), default=0))
            total_squared += sum(s * s for s in samples)

    if count == 0 or peak == 0:
        raise ValueError("디코딩한 음성 신호가 완전한 무음입니다. 원본 파일을 확인해 주세요")
    rms = math.sqrt(total_squared / count) / 32768.0
    dbfs = 20 * math.log10(max(rms, 1e-12))
    reporter(
        "audio",
        codec=codec,
        duration=round(duration, 1),
        rms_dbfs=round(dbfs, 1),
        message=f"오디오 정상 디코딩 · {duration:.1f}초 · 음량 {dbfs:.1f} dBFS",
    )
    if dbfs < -55:
        reporter("warning", message="음량이 매우 작습니다. 인식 결과가 없으면 원본 녹음 음량을 확인해 주세요")
    return {"codec": codec, "duration": duration, "rms_dbfs": dbfs}


def _recognize(
    model: Any,
    source: Path,
    *,
    language: str,
    use_vad: bool,
    reporter: Callable[..., None],
) -> tuple[str, str]:
    segments, info = model.transcribe(
        str(source),
        language=None if language == "auto" else language,
        beam_size=5,
        vad_filter=use_vad,
        task="transcribe",
    )
    duration = max(float(info.duration or 0), 0.01)
    pieces = []
    for segment in segments:
        pieces.append(segment.text)
        percent = max(1, min(99, int(float(segment.end) / duration * 100)))
        reporter("progress", percent=percent, message=f"음성 인식 중 · {percent}%")
    return "".join(pieces).strip(), str(info.language)


def run_transcription(
    input_file: Path,
    output_file: Path,
    *,
    language: str = "auto",
    vad: bool = True,
    model_factory: Callable[..., Any] | None = None,
    reporter: Callable[..., None] = emit,
    audio_preprocessor: Callable[..., dict[str, Any]] = prepare_audio,
) -> dict[str, Any]:
    """Transcribe with VAD recovery; never create a misleading empty TXT."""
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if language not in {"auto", "ko", "en", "ja", "zh"}:
        raise ValueError("Unsupported recognition language")
    # Remove partial results from a previous failed invocation.
    output_file.unlink(missing_ok=True)
    if model_factory is None:
        from faster_whisper import WhisperModel
        model_factory = WhisperModel

    with tempfile.TemporaryDirectory(prefix="whispercolab-") as folder:
        normalized = Path(folder) / "audio.wav"
        reporter("status", message="음성 파일 검사 및 디코딩 중")
        audio = audio_preprocessor(input_file, normalized, reporter=reporter)
        reporter("status", message="Whisper Large-v3 모델 준비 중 (최초 실행 시 다운로드)")
        model = model_factory("large-v3", device="cuda", compute_type="int8_float16")
        reporter("status", message="음성 인식 시작")
        text, detected_language = _recognize(
            model, normalized, language=language, use_vad=vad, reporter=reporter,
        )
        retried_without_vad = False
        if not text and vad:
            retried_without_vad = True
            reporter("warning", message="무음 필터 결과가 비어 있어 VAD를 끄고 자동 재시도합니다")
            text, detected_language = _recognize(
                model, normalized, language=language, use_vad=False, reporter=reporter,
            )

    if not text:
        raise RuntimeError(
            "음성 인식 결과가 비어 있습니다. 오디오 트랙은 디코딩됐지만 "
            "Whisper가 말을 인식하지 못했습니다. 녹음 음량·내용을 확인하거나 "
            "언어를 '자동 감지'로 바꿔 다시 실행해 주세요. "
            f"(오디오 {audio['duration']:.1f}초 / {audio['rms_dbfs']:.1f} dBFS)"
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text + "\n", encoding="utf-8")
    reporter(
        "done",
        percent=100,
        language=detected_language,
        characters=len(text),
        retried_without_vad=retried_without_vad,
    )
    return {
        "language": detected_language,
        "characters": len(text),
        "output": str(output_file),
        "retried_without_vad": retried_without_vad,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Whisper Large-v3 on Colab GPU")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", choices=("auto", "ko", "en", "ja", "zh"), default="auto")
    parser.add_argument("--no-vad", action="store_true")
    args = parser.parse_args()
    try:
        run_transcription(args.input, args.output, language=args.language, vad=not args.no_vad)
    except Exception as exc:
        emit("error", message=f"{type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
