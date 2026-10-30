"""Colab-native Whisper Large-v3 transcription with a lecture-quality preset.

No public server or external transcription API. Decode input using FFmpeg,
then transcribe with mild repetition mitigation and conservative VAD.
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
    """Extract the first audio stream as 16 kHz mono PCM and measure its level."""
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
        reporter("warning", message="음량이 매우 작습니다. 인식 결과가 없으면 원본 녹음을 확인하세요")
    return {"codec": codec, "duration": duration, "rms_dbfs": dbfs}


def transcribe_options(language: str, use_vad: bool, hotwords: str) -> dict[str, Any]:
    """Conservative lecture settings; never delete text just because it repeats.

    Repetitions may reflect genuine speech. Mitigate decoder loops without
    blindly removing adjacent, similar sentences from the TXT output.
    """
    return {
        "language": None if language == "auto" else language,
        "task": "transcribe",
        "beam_size": 5,
        "best_of": 5,
        "vad_filter": use_vad,
        # Preserve Whisper's conservative built-in VAD configuration.
        "condition_on_previous_text": False,
        "repetition_penalty": 1.10,
        # Some lecture speakers really repeat words; do not forbid all 3-grams.
        "no_repeat_ngram_size": 0,
        "temperature": (0.0, 0.2, 0.4, 0.6),
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "hotwords": hotwords or None,
    }


def _recognize(
    model: Any,
    source: Path,
    *,
    language: str,
    use_vad: bool,
    hotwords: str,
    reporter: Callable[..., None],
) -> tuple[str, str]:
    segments, info = model.transcribe(
        str(source), **transcribe_options(language, use_vad, hotwords)
    )
    duration = max(float(info.duration or 0), 0.01)
    paragraphs: list[str] = []
    current: list[str] = []
    paragraph_start: float | None = None
    previous_end: float | None = None
    uncertain = 0
    total_segments = 0
    for segment in segments:
        total_segments += 1
        text = segment.text.strip()
        start = float(segment.start)
        end = float(segment.end)
        if text:
            # Group the literal decoded words into readable lecture paragraphs.
            # No word replacement/deduplication is applied.
            if current and (
                (previous_end is not None and start - previous_end >= 1.8)
                or (paragraph_start is not None and start - paragraph_start >= 45)
            ):
                paragraphs.append(" ".join(current))
                current = []
                paragraph_start = None
            if paragraph_start is None:
                paragraph_start = start
            current.append(text)
            previous_end = end
        avg_logprob = getattr(segment, "avg_logprob", None)
        compression = getattr(segment, "compression_ratio", None)
        if ((avg_logprob is not None and avg_logprob < -1.0)
                or (compression is not None and compression > 2.4)):
            uncertain += 1
        progress = max(1, min(99, int(end / duration * 100)))
        reporter("progress", percent=progress, message=f"음성 인식 중 · {progress}%")
    if current:
        paragraphs.append(" ".join(current))
    if uncertain:
        reporter(
            "warning",
            message=f"신뢰도가 낮을 수 있는 구간 {uncertain}/{total_segments}개. 결과를 원본과 대조해 주세요.",
        )
    return "\n\n".join(paragraphs).strip(), str(info.language)


def run_transcription(
    input_file: Path,
    output_file: Path,
    *,
    language: str = "auto",
    vad: bool = True,
    hotwords: str = "",
    compute_type: str = "float16",
    model_factory: Callable[..., Any] | None = None,
    reporter: Callable[..., None] = emit,
    audio_preprocessor: Callable[..., dict[str, Any]] = prepare_audio,
) -> dict[str, Any]:
    """Transcribe with conservative VAD; retry only when the result is blank."""
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if language not in {"auto", "ko", "en", "ja", "zh"}:
        raise ValueError("Unsupported recognition language")
    if compute_type not in {"float16", "int8_float16"}:
        raise ValueError("Unsupported GPU precision")
    if len(hotwords) > 500:
        raise ValueError("전문용어 힌트는 500자 이하로 입력해 주세요")
    output_file.unlink(missing_ok=True)
    if model_factory is None:
        from faster_whisper import WhisperModel
        model_factory = WhisperModel

    with tempfile.TemporaryDirectory(prefix="whispercolab-") as folder:
        normalized = Path(folder) / "audio.wav"
        reporter("status", message="음성 파일 검사 및 디코딩 중")
        audio = audio_preprocessor(input_file, normalized, reporter=reporter)
        reporter("status", message="Whisper Large-v3 모델 준비 중 (최초 실행 시 다운로드)")
        model = model_factory("large-v3", device="cuda", compute_type=compute_type)
        reporter("status", message="음성 인식 시작 · 반복 방지 / 이전 문맥 이어받기 OFF")
        text, detected_language = _recognize(
            model,
            normalized,
            language=language,
            use_vad=vad,
            hotwords=hotwords,
            reporter=reporter,
        )
        retried_without_vad = False
        if not text and vad:
            retried_without_vad = True
            reporter("warning", message="무음 필터 결과가 비어 있어 VAD를 끄고 자동 재시도합니다")
            text, detected_language = _recognize(
                model,
                normalized,
                language=language,
                use_vad=False,
                hotwords=hotwords,
                reporter=reporter,
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
    parser.add_argument("--hotwords", default="", help="Comma-separated vocabulary hints")
    parser.add_argument("--compute-type", choices=("float16", "int8_float16"), default="float16")
    args = parser.parse_args()
    try:
        run_transcription(
            args.input,
            args.output,
            language=args.language,
            vad=not args.no_vad,
            hotwords=args.hotwords,
            compute_type=args.compute_type,
        )
    except Exception as exc:
        emit("error", message=f"{type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
