# WhisperColab

**Whisper Large-v3**를 Google Colab 무료 GPU에서 실행하는 **노트북 내장 음성→TXT 변환기**입니다.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dariring/WhisperColab/blob/main/WhisperColab.ipynb)

## 빠른 시작

1. 위 **Open in Colab**을 클릭합니다.
2. 메뉴에서 **런타임 → 런타임 유형 변경 → T4 GPU**를 선택합니다. GPU 할당은 보장되지 않습니다.
3. **런타임 → 모두 실행**을 클릭합니다.
4. 아래 셀에 나타나는 **음성 파일 선택** 버튼으로 오디오를 올립니다. **선택 즉시 자동 변환**을 시작합니다.
5. 진행률과 변환 결과를 확인한 후 **TXT 다운로드** 버튼을 누릅니다.

**터널/서버 설정, 별도 웹 링크, 로그인 비밀번호, OpenAI API 키는 필요하지 않습니다.** 모든 작업은 Colab 노트북 UI 내부에서 이루어집니다.

## 기능

- OpenAI Whisper **Large-v3** 원본 모델 가중치 기반 음성 인식 (Turbo 아님)
- `faster-whisper` / `CTranslate2` 기반 NVIDIA GPU `int8_float16` 추론
- Colab 내부 `ipywidgets` UI: 업로드, 언어 선택, 무음 구간 제외 설정, 진행률, TXT 미리보기·다운로드
- 파일 선택 시 자동 처리, 별도 시작 버튼 불필요
- 자동 언어 감지 또는 한국어·영어·일본어·중국어 선택
- MP3, WAV, M4A, FLAC, OGG, OPUS, AAC, MP4, WEBM, WMA, AIFF, MKV 지원
- 파일 크기 **75MB 제한** (노트북 브라우저 업로드 한계는 별도)
- 처리 완료 후 업로드 원본 삭제, 변환 결과는 런타임의 `/content/whispercolab/results/`에 저장
- 처음 이용 시 공개된 모델 가중치를 다운로드합니다. 재변환 시 모델을 새로 불러오기 때문에 잠시 시간이 걸릴 수 있습니다.

## 구조

| 파일 | 설명 |
|---|---|
| `WhisperColab.ipynb` | Colab GPU 설치 및 노트북 UI |
| `whisper_worker.py` | GPU 음성 변환 워커 + JSON 진행률 출력 |
| `requirements.txt` | 음성 인식, NVIDIA 라이브러리, ipywidgets |
| `tests/test_worker.py` | GPU 없이 실행하는 기능·구조 테스트 |

노트북 UI 위젯은 별도의 Python GPU 워커 프로세스를 호출합니다. 이 프로세스는 CUDA 12/cuDNN 9 동적 라이브러리 경로를 전달받아 실행되며, 외부 웹서버를 띄우지 않습니다.

## 문제 해결

- **GPU 없음:** 런타임 유형 변경에서 T4 GPU를 선택하세요. 할당량이 없다면 나중에 다시 시도해야 합니다.
- **업로드 버튼이 나타나지 않음:** 노트북의 설치 셀과 UI 셀을 다시 실행하세요.
- **음성 인식 실패:** UI에 표시된 오류를 확인하세요. 메모리나 VRAM이 부족한 경우 더 짧은 파일로 시험하세요.
- **다운로드 버튼이 동작하지 않음:** 마지막 `STEP 3 (선택)` 셀을 실행하거나 왼쪽 파일 탐색기에서 `/content/whispercolab/results` 폴더의 TXT를 다운로드하세요.
- **긴 오디오:** 무료 Colab은 세션 시간을 보장하지 않습니다. 변환이 끝나는 즉시 TXT를 내려받으세요.

## 무료 Colab 정책

Colab은 무료 관리형 런타임에서 **노트북 UI를 우회하여 외부 웹 UI를 주요 조작 수단으로 사용하는 방식**을 제한합니다. 이 저장소는 해당 구조를 제거하고, **Colab 자체 노트북 UI**만 사용하도록 변경했습니다. 그렇더라도 무료 GPU 할당·사용 시간 제한은 그대로 적용되며 작업이 조기에 종료될 가능성은 남아 있습니다.

- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)

> **보안:** 파일은 외부 Whisper API로 전송하지 않으며 Colab 런타임에 업로드되어 처리됩니다. 모델 다운로드와 Colab 서비스 자체 통신은 발생할 수 있습니다.
