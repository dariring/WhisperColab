# WhisperColab

Google Colab 무료 GPU에서 OpenAI **Whisper Large-v3**를 실행하는 노트북 기반 음성→TXT 변환기입니다.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dariring/WhisperColab/blob/main/WhisperColab.ipynb)

## 사용 방법

1. **Open in Colab**을 누르고 **런타임 → 런타임 유형 변경 → T4 GPU**를 설정하세요.
2. **STEP 1** 셀의 **▶ 실행**을 눌러 설치하세요.
3. **STEP 2**에서 언어·무음 구간 제외를 선택하고 **▶ 실행**을 누르세요.
4. **Colab 기본 파일 업로드 창**에서 MP3, WAV, M4A 등을 하나 선택하세요.
5. 파일이 업로드되면 자동으로 **Whisper Large-v3 변환**이 시작되고, 완료 시 **TXT가 자동 다운로드**됩니다.
6. 다음 파일을 변환하려면 **STEP 2 셀만 다시 실행**하세요.

### 기존 화면에서 '음성 파일 선택 (1)'만 표시되고 멈췄다면

기존 버전의 \`ipywidgets.FileUpload\`가 일부 Colab 환경에서 파일을 Python 커널에 전달하지 못해 업로드 개수만 표시하던 문제였습니다. **새 노트북을 다시 열고 STEP 1, STEP 2를 차례로 실행하세요.** 새로운 버전은 위젯의 비동기 업로드 이벤트를 아예 사용하지 않고, 공식 **\`google.colab.files.upload()\`**로 실행 중인 셀에서 업로드합니다.

Colab 업로드 창까지 작동하지 않는 환경에서는 **왼쪽 파일 탭**에서 파일을 업로드한 후, STEP 2에서 **파일 가져오기 → Colab 파일 경로 사용**을 선택해 해당 경로(\`/content/...\`)를 입력하면 됩니다.

### 주요 기능

- OpenAI Whisper **Large-v3** 가중치 기반, \`faster-whisper\` / \`CTranslate2\`, GPU \`int8_float16\`
- Colab **노트북 기본 폼 UI**로 언어 선택 및 VAD 무음 구간 필터 조절
- **업로드 후 자동 변환**, 진행률 표시, 결과 미리보기, TXT 자동 다운로드
- 75MB 이하 파일 지원: MP3, WAV, M4A, FLAC, OGG, OPUS, AAC, MP4, WEBM, WMA, AIFF, MKV
- 변환 결과: \`/content/whispercolab/results/\` (STEP 3에서 다시 다운로드 가능)
- 외부 웹서버, TryCloudflare, OpenAI API 키 불필요

### 유의사항

- Colab 무료 GPU 할당과 세션 지속 시간은 보장되지 않습니다. 작업 종료 후 TXT를 내려받으세요.
- 최초 실행 시 모델 가중치를 다운로드하므로 다음 실행보다 더 오래 걸릴 수 있습니다.
- 파일은 외부 Whisper API로 보내지 않고 Colab 런타임에서 변환합니다. 다만 Colab의 기본 업로드 시스템을 통해 전달됩니다.

관련 코드: [WhisperColab.ipynb](./WhisperColab.ipynb), [whisper_worker.py](./whisper_worker.py).
