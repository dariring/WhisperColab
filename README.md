# WhisperColab

OpenAI의 **Whisper Large-v3**로 음성을 텍스트로 변환하는 Colab 웹패널입니다. **Google Colab 무료 GPU + TryCloudflare Quick Tunnel**을 사용합니다.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dariring/WhisperColab/blob/main/WhisperColab.ipynb)

## 사용법

1. 위 **Open in Colab** 버튼을 클릭합니다.
2. Colab에서 **런타임 → 런타임 유형 변경 → T4 GPU** (사용 가능한 경우) 선택
3. **런타임 → 모두 실행**을 누릅니다.
4. 실행 결과의 `https://...trycloudflare.com` 주소를 열고 로그인합니다.
   - **사용자명:** `whisper`
   - **비밀번호:** 노트북 실행 결과에 생성되는 임시 비밀번호
5. MP3, WAV, M4A 등의 파일을 선택하면 **자동으로 변환**이 시작됩니다.
6. 진행 상태를 확인하고 **TXT 다운로드** 버튼을 누릅니다.

> Colab 노트북은 매번 실행할 때 주소와 비밀번호를 새로 만듭니다. Colab 런타임이 종료되면 웹패널도 종료됩니다.

## 주요 기능

- **Whisper Large-v3** 원본 모델 가중치 기반 음성 인식. OpenAI API 호출 또는 API 키 불필요
- `faster-whisper` + `CTranslate2` 기반 CUDA GPU `int8_float16` 추론 (Colab T4의 VRAM 절약)
- 모바일/PC 반응형 한국어 웹패널, 드래그 앤 드롭 업로드
- 파일 선택 즉시 자동 실행, 언어 자동 감지 / 한국어·영어·일본어·중국어 고정 선택
- VAD 무음 구간 제외 옵션, 작업 상태 및 진행률 표시
- 변환 결과 확인 / 텍스트 복사 / UTF-8 TXT 다운로드
- 패널 전체 HTTP Basic 인증, 최대 **75MB** 파일 업로드, GPU 작업 순차 처리
- 원본 파일은 처리 완료 시 삭제하고 TXT는 최대 **1시간** 임시 보관

## 구성

| 파일 | 역할 |
| --- | --- |
| `WhisperColab.ipynb` | GPU 확인, 패키지 설치, 서버와 TryCloudflare 터널 실행 |
| `app.py` | 음성 인식 API, 업로드 처리, 단일 GPU 작업 대기열, 인증 |
| `static/index.html` | 프론트엔드 웹패널 (HTTP 폴링) |
| `requirements.txt` | Python 의존성 |

**진행 상태는 HTTP 폴링 방식**으로 확인합니다. Cloudflare Quick Tunnel에서 SSE가 지원되지 않아 SSE/WebSocket에 의존하는 대기열 대신 일반 HTTP 요청을 사용합니다.

## 무료 티어 및 보안 주의사항

- **Colab 무료 GPU는 항상 제공되지 않습니다.** GPU 종류, 사용 한도, 세션 길이도 변동되며, 무료 Colab에서 웹 UI 중심으로 사용하면 런타임이 조기에 종료될 수 있습니다. 24시간 공개 서비스에 사용하지 마세요. [Colab 공식 FAQ](https://research.google.com/colaboratory/faq.html)
- **TryCloudflare는 임시 개발용 터널**입니다. URL이 매번 바뀌며 가동 시간이 보장되지 않습니다. 링크를 아는 모든 사람이 연결을 시도할 수 있으므로 임시 비밀번호를 외부에 공유하지 마세요. [Cloudflare Quick Tunnels 안내](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)
- 오디오는 OpenAI API에 전달하지 않고 **현재 Colab 런타임에서 처리**됩니다. 다만 브라우저에서 Colab까지 파일을 전송할 때 Cloudflare 터널을 경유합니다.
- 75MB는 이 앱의 제한입니다. 네트워크나 Cloudflare 자체 제한에 따라 더 작은 파일에서도 업로드가 실패할 수 있습니다.
- 음성이 매우 길면 무료 GPU·세션 제한으로 변환에 실패할 수 있습니다. 중요한 원본 파일은 반드시 따로 보관하세요.

## 개발 환경에서 직접 실행

Python 3.10+와 NVIDIA CUDA 12/cuDNN 9 환경에서:

```bash
pip install -r requirements.txt
export PANEL_PASSWORD="replace-with-a-strong-password"
uvicorn app:app --host 127.0.0.1 --port 7860
```

접속: `http://127.0.0.1:7860` · 사용자명 `whisper`. GPU 추론에는 CUDA 12 및 cuDNN 9 라이브러리 경로 설정이 필요합니다. Colab 노트북은 이를 자동 처리합니다.

모델: [OpenAI Whisper Large-v3](https://github.com/openai/whisper) · 추론 구현: [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
