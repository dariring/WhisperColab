"""End-to-end M4A/AAC audio decoding tests; no neural model/GPU needed."""
import shutil
import subprocess
import wave
from array import array
from math import pi, sin
from pathlib import Path

import pytest
from whisper_worker import prepare_audio


pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="Requires the ffmpeg/ffprobe binaries available on Google Colab",
)


@pytest.fixture
def m4a_sample(tmp_path):
    wav_path = tmp_path / "test.wav"
    samples = array("h", [int(10000 * sin(2 * pi * i * 440 / 16000)) for i in range(16000)])
    with wave.open(str(wav_path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(samples.tobytes())
    m4a = tmp_path / "sample.m4a"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(wav_path), "-c:a", "aac", "-b:a", "128k", str(m4a)],
        check=True,
    )
    return m4a


def test_aac_m4a_can_be_converted_and_measured(m4a_sample, tmp_path):
    destination = tmp_path / "prepared.wav"
    events = []
    info = prepare_audio(
        m4a_sample, destination,
        reporter=lambda kind, **kw: events.append((kind, kw)),
    )
    assert destination.is_file()
    assert info["codec"] == "aac"
    assert 0.9 <= info["duration"] <= 1.2
    assert info["rms_dbfs"] > -30
    with wave.open(str(destination), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 16000
        assert wav.getsampwidth() == 2
    assert any(kind == "audio" for kind, _ in events)


def test_silent_m4a_reports_silence(tmp_path):
    sample = tmp_path / "silent.wav"
    with wave.open(str(sample), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b"\x00\x00" * 16000)
    destination = tmp_path / "silent.m4a"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(sample),
         "-c:a", "aac", str(destination)],
        check=True,
    )
    with pytest.raises(ValueError, match="완전한 무음"):
        prepare_audio(destination, tmp_path / "decoded.wav")
