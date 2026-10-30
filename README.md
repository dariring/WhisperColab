# WhisperColab

Google Colab 무료 GPU에서 OpenAI **Whisper Large-v3**를 실행하는 노트북 기반 음성→TXT 변환기입니다.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dariring/WhisperColab/blob/main/WhisperColab.ipynb)

## 2026-10-30 오류 수정: `metadata_errors`

`TypeError: open() got an unexpected keyword argument 'metadata_errors'`는 `faster-whisper 1.2.1`과 `PyAV 18+` 간 API 호환성 문제입니다. 현재는 **PyAV 17.1.0**을 사용하도록 고정했고, 설치 단계에서 WAV 디코딩 호환성을 자동 검증합니다.

**이미 실행 중인 Colab 세션에서 발생했다면:** 노트북을 새로 열어 **STEP 1 설치 셀을 다시 실행**한 다음 **STEP 2에서 파일을 다시 올려** 주세요. 현재 런타임을 그대로 사용해도 됩니다. 복구되지 않으면 런타임을 재시작하고 STEP 1부터 실행해 주세요.

`Warning: You are sending unauthenticated requests to the HF Hub`는 **공개 모델을 로그인 없이 내려받을 때 발생하는 경고**이며 위 오류의 원인이 아닙니다. 공개 모델 이용을 위해 HF_TOKEN을 발급받을 필요는 없습니다.

## M4A 변환 결과가 빈 TXT로 나오는 문제 (2026-10-30 수정)

이전 버전은 Whisper가 음성 구간을 하나도 인식하지 못했을 때도 **빈 TXT를 성공 결과로 처리**했습니다. 정확한 원인은 입력 M4A와 실행 로그가 있어야 확인할 수 있지만, 음성 구간 검출기(VAD)가 음성을 제외하거나 실제 오디오 신호가 누락된 경우에 발생할 수 있습니다.

새 버전:
- **M4A/AAC 등 오디오 트랙을 ffprobe로 확인한 후, FFmpeg로 16kHz 모노 PCM WAV로 디코딩**하여 Whisper에 전달합니다.
- **디코딩된 음성의 길이와 RMS 음량(dBFS)을 표시**하고, 완전한 무음이면 즉시 오류를 표시합니다.
- VAD를 켠 경우 **문자 인식 결과가 비면 VAD 없이 자동으로 한 번 재시도**합니다.
- 그래도 인식 결과가 비면 **실패로 처리하고 빈 TXT 다운로드를 막습니다.**
- 긴 무음에서 반복 환각을 줄일 수 있도록 **무음 구간 제외는 기본 ON**입니다. VAD로 모든 음성이 제외되면 OFF로 자동 재시도합니다.

**이미 Colab을 열어둔 상태라면:** 새 노트북을 열고 **STEP 1을 다시 실행해 코드를 갱신한 뒤**, STEP 2에서 M4A를 다시 업로드하세요. STEP 2에서 'Colab 파일 경로 사용'을 선택하여 이전 파일의 경로를 입력해도 됩니다(원본이 남아 있는 경우). 오류 발생 시 STEP 2에 표시되는 오디오 길이·음량·오류 메시지를 확인해 주세요.

## 강의 전사 인식 품질 개선

원본 녹음 없이 전사 결과만으로 오류 원인을 확정할 수는 없지만, 앞 문장의 반복이나 `아멘 아멘...` 같은 환각은 긴 강의에서 특히 무음·잡음·문맥 누적과 관련될 수 있습니다.

- **이전 문맥 이어받기 OFF:** `condition_on_previous_text=False`로 이전 구간의 잘못된 문장이 다음 구간에 반복되는 현상을 줄입니다.
- **기본 VAD ON:** Whisper가 긴 무음에서 말을 만들어내는 현상을 줄일 수 있도록 보수적인 기본 VAD를 적용합니다. 완전히 비면 VAD OFF 자동 재시도 기능은 유지합니다.
- **완만한 반복 페널티:** `repetition_penalty=1.10`; 실제 발화가 반복될 수도 있으므로 결과 텍스트를 기계적으로 삭제하지 않습니다.
- **FP16 정확도 우선:** Colab T4에서 기본 `float16`을 사용합니다. GPU 메모리가 부족하면 `int8_float16` 선택이 가능합니다.
- **전문용어 힌트:** STEP 2의 `전문용어_힌트`에 `공자, 안회, 자로, 자공, 논어, 맹자` 같은 단어를 입력할 수 있습니다. 일반 음성이라면 이 항목을 비우세요. 힌트가 실제 음성에 없는 단어를 유도할 수도 있으므로 원본과 비교해 확인하세요.
- **문단 나누기:** 인식된 구간을 간단한 시간/쉼표 기준으로 읽기 쉽게 나누되, 원문 문장을 임의로 제거하지 않습니다.

이 설정이 녹음 잡음, 발화 거리, 여러 사람의 동시 발화 등 입력 품질 문제까지 해결하는 것은 아닙니다. 개선 정도는 실제 오디오와 비교해야 합니다.

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

- OpenAI Whisper **Large-v3** 가중치 기반, \`faster-whisper\` / \`CTranslate2\`, GPU \`float16\` (기본) 또는 \`int8_float16\`
- Colab **노트북 기본 폼 UI**로 언어 선택 및 VAD 무음 구간 필터 조절
- **업로드 후 자동 변환**, 음성 길이/음량 검사, 빈 결과 재시도, 진행률 표시, 결과 미리보기, TXT 자동 다운로드
- 75MB 이하 파일 지원: MP3, WAV, M4A, FLAC, OGG, OPUS, AAC, MP4, WEBM, WMA, AIFF, MKV
- 변환 결과: \`/content/whispercolab/results/\` (STEP 3에서 다시 다운로드 가능)
- 외부 웹서버, TryCloudflare, OpenAI API 키 불필요

### 유의사항

- Colab 무료 GPU 할당과 세션 지속 시간은 보장되지 않습니다. 작업 종료 후 TXT를 내려받으세요.
- 최초 실행 시 모델 가중치를 다운로드하므로 다음 실행보다 더 오래 걸릴 수 있습니다.
- 파일은 외부 Whisper API로 보내지 않고 Colab 런타임에서 변환합니다. 다만 Colab의 기본 업로드 시스템을 통해 전달됩니다.

관련 코드: [WhisperColab.ipynb](./WhisperColab.ipynb), [whisper_worker.py](./whisper_worker.py).
