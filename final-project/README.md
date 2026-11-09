# 손동작과 음성으로 이용하는 주민센터 안내 키오스크

## 1. 프로젝트 소개

Jetson Orin Nano에서 카메라와 마이크로 사용자의 요청을 받아 주민등록등본 업무를 안내하는 온디바이스(On-Device) 비접촉 키오스크 프로젝트이다. 카메라에서 사람과 손동작 및 사물(카드·문서) 지시 여부를 인식하고, 사용자의 말을 문자로 변환한 뒤, 인식 결과를 하나의 문맥으로 합쳐 로컬 Gemma가 안내 문장을 생성하도록 구현했다.

공공기관 망분리 환경과 개인정보 보호를 고려하여 외부 클라우드 API 통신 없이 100% 로컬에서 동작하며, 모든 영상·음성 데이터는 디스크에 저장하지 않고 RAM에서만 휘발성으로 처리한다. 화면은 무거운 웹 브라우저 없이 파이썬 내장 Tkinter 전용 창으로 표시한다. 실행 코드는 `kiosk_jetson3.py` 한 파일에 통합했으며 모델 파일은 기존 `src/models` 경로에서 불러온다.

현재 구현 범위는 등본 안내 시연이다. L 모양은 프로젝트에서 정한 멀티모달 업무 선택 동작이며 일반적인 수어 번역 기능이 아니다. 카드·문서 후보를 감지하지만 신분증 종류·진위·글자 내용(OCR)은 판독하지 않는다. 본인 인증 및 실제 증명서 발급 시스템도 연결되어 있지 않다.

---

## 2. 동작 시나리오

1. 운영자가 프로그램을 실행하고 운영 시작을 누른다.
2. 카메라 영상이 나타나고, 사람이 감지되면 Piper TTS가 인사한다.
3. 사용자가 엄지와 검지로L 모양을 만들면 등본 요청으로 선택한다. 다른 손의 검지로 바닥(트레이)의 사각형 카드나 문서를 가리킬 수도 있다.
4. 사용자가 “이걸로 떼줘” 또는 “등본 뽑아줘”라고 말하고 2초간 침묵하면 입력을 마감한다.
5. 메모리 확보를 위해 비전 프로세스를 즉시 종료한 뒤, Whisper가 음성을 문자로 변환하고 프로그램이 손동작·문서 후보·음성 내용을 JSON 문맥으로 합친다.
6. Gemma가 문맥에 맞는 안내 문장을 생성하면 화면의 큰 글자와 스피커 음성으로 출력한다.
7. 사용자가 카메라에서 3초간 벗어나면 이전 대화 기록을 완전히 비우고(Stateless) 다음 이용을 위한 대기 상태로 돌아간다.

말하지 않고 L 동작만 선택한 경우에는 선택 확정 후 3초에 자동 제출한다. 입력 중 Enter 또는 Space를 보내는 물리 버튼으로도 즉시 제출할 수 있다. 초기 무음만으로는 제출하지 않으며, 25초 동안 말이나 L 선택이 없으면 입력을 취소한다.

---

## 3. 실행 환경과 파일 배치

-대상 장치: NVIDIA Jetson Orin Nano, Ubuntu 데스크톱 (통합 메모리 아키텍처)
-연결 장치: 모니터, 카메라, 마이크, 스피커
-Python: 기존 Jetson 가상환경 사용, Python 3.10 문법 호환 확인
-비전: OpenCV, NumPy, MediaPipe Tasks API, YOLOv11 (TensorRT `int8.engine`)
-답변 생성: `llama-cpp-python` 및 Gemma 4비트 양자화 GGUF 모델 (`Q4_K_M`)
-음성 인식: `whisper.cpp` 실행 파일(`whisper-cli`) 및 모델
-음성 합성: Piper 및 한국어 ONNX 모델 (독립 가상환경 `.piper_venv` 분리 구동 지원)
-화면: Tkinter
-녹음·재생: ALSA의 `arecord`, `aplay`

자동 패키지 설치, 버전 변경, 모델 재학습은 수행하지 않는다. Pillow, PyAudio, sounddevice는 이 코드에서 직접 사용하지 않는다.

`kiosk_jetson3.py`와 이 README는 `~/vision-llm`에 두고 기존 `src`를 유지한다.

| 항목 | 기본 경로 |
|---|---|
| 실행 파일 | `~/vision-llm/kiosk_jetson3.py` |
| 비전·Gemma Python | 파일을 실행한 Python, 예: `.venv/bin/python` |
| Piper Python | `.piper_venv/bin/python`, 없으면 실행한 Python |
| 손 모델 | `src/models/MediaPipe/hand_landmarker.task` |
| 사람 감지 모델 | `src/models/MediaPipe/pose_landmarker_lite.task` 또는 `pose_landmarker_full.task` |
| Gemma | `src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf` |
| Piper | `src/models/Piper/ko_KR-kss-medium.onnx` |
| Piper 설정 | `src/models/Piper/ko_KR-kss-medium.onnx.json` |
| Whisper 실행 파일 | `whisper.cpp/build-cpu/bin/whisper-cli` 또는 `whisper.cpp/build/bin/whisper-cli` |
| Whisper 모델 | `whisper.cpp/models/ggml-base.bin` |
| YOLO 엔진 | `src/models/YOLO/yolo11n_int8.engine` |

Pose는 Lite 파일이 있으면 우선 사용하고, 없으면 Full 파일을 사용한다. Whisper는 `src/models/whisper.cpp` 아래 경로도 탐색한다. 모델 경로는 Python 파일의 위치를 기준으로 결정한다.

---

## 4. 실행 방법

Jetson에 연결된 모니터의Ubuntu 데스크톱 터미널에서 실행한다. SSH 접속만으로 로컬 화면 세션이 만들어지는 것은 아니다.

### 환경 검사

```bash
cd ~/vision-llm
.venv/bin/python kiosk_jetson3.py --check
```

모델 파일, Python 패키지 import, GPU 오프로딩 지원, Tkinter 및 오디오 명령 존재 여부를 일괄 검사한다. 이 검사는 실제 모델 추론이나 카메라 연결을 실행하지 않으므로 파일이 있다는 결과만으로 실제 동작이 보장되지는 않는다.

### 카메라 검사

```bash
.venv/bin/python kiosk_jetson3.py --diagnose-camera
```

카메라를 열어 첫 프레임의 수신 여부와 크기를 확인한다. 영상은 프로그램 창의카메라만 테스트 버튼으로도 확인할 수 있다. 이 버튼은 사람·손 모델 없이도 카메라를 검사한다.

### 마이크·스피커 검사

```bash
.venv/bin/python kiosk_jetson3.py --list-audio
.venv/bin/python kiosk_jetson3.py --diagnose-audio
```

두 번째 명령은 실제로 3초 동안 녹음한 뒤 스피커로 즉시 재생한다. 진단용 임시 녹음은 재생 후 삭제한다. 다른 프로그램이 카메라나 마이크를 사용 중이면 먼저 종료한다.

### 실행

```bash
# Gemma 없이 입력 흐름을 먼저 확인
.venv/bin/python kiosk_jetson3.py --no-llm

# 전체 화면으로 정상 실행
.venv/bin/python kiosk_jetson3.py --fullscreen
```

두 명령은 각각 별도로 실행한다. 첫 프로그램을 종료한 뒤 다음 명령을 실행한다. `--no-llm`은 Gemma만 생략하고 기본 안내 문장을 사용하므로 카메라·손동작·Whisper·Piper는 준비된 경우 계속 동작한다.

자동 운영 시작:

```bash
.venv/bin/python kiosk_jetson3.py --fullscreen --autostart
```

| 조작 | 기능 |
|---|---|
| 운영 시작 | 카메라 및 사람 감지 시작 |
| 카메라만 테스트 | 모델 없이 영상 확인 |
| 운영 중지 | 장치와 작업 프로세스 정리 |
| F11 | 전체 화면 전환 |
| Esc | 전체 화면 해제 |
| Ctrl+Q | 종료 |
| Enter / Space | 입력 중 즉시 제출 |

---

## 5. 주요 구현 방법과 아키텍처

### 멀티 프로세스 분리와 인메모리 파이프 통신 (IPC)

단일 스크립트(`kiosk_jetson3.py`)가 `--worker` 플래그를 통해 자기 자신을 분기 호출하여 총 5개의 독립된 OS 프로세스(Main GUI, Vision, STT, LLM, TTS)로 동작한다.

-표준 입출력(`stdin`/`stdout`) 메모리 파이프: 프로세스 간 데이터 교환 시 디스크 I/O 병목과 개인정보 잔류를 막기 위해 임시 파일을 쓰지 않고 리눅스 커널 파이프를 통해 RAM 상에서만 통신한다.
-`KIOSK_RESULT:` 프리픽스 JSON 프로토콜: C++ 기반 AI 라이브러리들의 시스템 로그와 실제 결과 데이터를 구분하기 위해, 워커 프로세스는 결과 딕셔너리를 한 줄의 JSON 문자열로 직렬화하고 앞에 `KIOSK_RESULT:` 접두사를 붙여 출력(`emit()`)한다. 메인 프로세스는 이 접두사가 붙은 줄만 파싱(`parse_result()`)한다.

### 카메라 캡처(최신 프레임 단일 버퍼링)와 사람 감지

`CameraReader`는 OpenCV로 카메라를 열고 별도 데몬 스레드(`_read()`)에서 프레임을 비동기로 계속 읽는다.
-최신 프레임 단일 버퍼링: V4L2 버퍼 크기 최소화(`CAP_PROP_BUFFERSIZE, 1`) 및 GStreamer(`drop=true max-buffers=1`) 설정과 함께, 큐에 과거 프레임을 쌓아두지 않고 단일 변수(`self.frame`)에 최신 프레임 1장만 덮어써서 보관한다. 이를 통해 AI 추론 속도와 무관하게 인풋 렉(Latency) 없는 실시간 화면을 공급한다.

`vision_worker()`는 비전 전용 프로세스에서 실행된다. 영상의 긴 변을 최대 640픽셀로 줄이고 BGR을 연속된 RGB 배열로 변환하여 MediaPipe에 전달한다.
-조건부 계층 추론 및 스로틀링: 상체 감지(MediaPipe Pose 또는 YOLO person 클래스)는 연산 부하를 줄이기 위해 0.4초(400ms) 간격으로만 실행하며, 사람이 감지되고 입력 대기 상태일 때만 손 모델(최대 2개)을 구동한다.
- 추론이 끝난 후에만 원본 프레임 위에 손 관절과 문서 후보 박스를 그리고 가로 512픽셀로 축소한 뒤, 메모리 상에서 PNG로 압축(`cv2.imencode`) 및 Base64 텍스트로 변환해 메인 GUI 프로세스로 스트리밍한다. GUI 역시 `maxsize=1` 큐와 Tkinter `PhotoImage`를 사용해 약 $8 \sim 10\text{ FPS}$로 지연 없이 렌더링한다.

### 손동작 판별 (`hand_signs`)

`hand_signs()`는 21개 손 관절 좌표로 관절 간 벡터 내적(각도)과 유클리드 거리를 계산한다. 손목과 중지 시작 관절 사이의 거리를 기준으로 정규화하여 손 크기의 영향을 줄인다.

- 검지만 펴고 엄지가 충분히 벌어져 있으며 엄지·검지 방향 사이 각도가 $45^\circ \sim 135^\circ$이면L 동작으로 판정한다.
- 검지만 펴고 엄지가 벌어지지 않은 경우에는가리키기(Pointing) 동작으로 판정하며, 검지 두 번째 마디에서 손끝으로 향하는 단위 방향 벡터(`direction`)를 추출한다.

`Hold.update()`는 1~2프레임의 순간적인 오인식으로 선택되지 않도록 지속 시간을 확인한다. L 선택은 기본적으로 0.7초 이상, 3개 이상의 연속 관측이 필요하다.

### 카드·문서 후보 감지 및 광선 교차 판정 (`document_target` & `ray_box`)

`document_target()`는 화면에서 가리키는 손가락(`pointers`)이 존재할 때만 조건부로 실행되어 바닥의 카드나 문서 영역을 찾는다.

1.사각 후보 탐지 (YOLO TensorRT / 윤곽선 검출): TensorRT로 가속된 `yolo11n_int8.engine` 모델 또는 OpenCV 경계선·다각형 근사(Grayscale $\rightarrow$ Gaussian Blur $\rightarrow$ Canny $\rightarrow$ Morphological Close $\rightarrow$ Contour)를 통해 화면 내 카드·문서 후보의 바운딩 박스(`[x1, y1, x2, y2]`)를 검출한다.
2.Ray-casting (`ray_box`): 손끝 좌표에서 가리키는 방향 벡터로 뻗어 나가는 가상의 반직선(Ray)을 쏘아, 후보 박스와 교차하는지 슬랩(Slab) 알고리즘으로 계산한다. 손끝 뒤쪽의 물체는 무시하고 유효 거리(`scale 3`) 내에서 가장 정교하게 교차하는 박스를 선택한다.
3.시간적 필터링 (`box_iou`): 손 떨림이나 조명 노이즈를 막기 위해 연속 프레임에서 후보 박스의 겹침 비율($\text{IoU} \ge 0.25$)이 유지된 상태로 0.4초(최소 3프레임) 이상 지속되어야만 최종 문서 지시(`pointed_document_candidate = True`)로 확정한다.

### 녹음과 자동 제출 (VAD 및 무결성 검사)

`Microphone`은 `arecord`를 별도 프로세스로 실행하여 16kHz, 모노, PCM16 음성을 받아 RAM에 보관한다. 리눅스 `select.select` 시스템 콜로 0.1초 단위로 오디오 파이프를 감시하며, 4초 이상 스트림이 들어오지 않으면 즉시 하드웨어 연결 오류로 안전하게 종료한다.

`VoiceActivity.feed()`는 20ms 단위 PCM 블록의 RMS 음량을 계산한다. 순간적인 소음에 반응하지 않도록 기준 음량(`vad_threshold`) 이상이120ms(6블록) 이상 연속 지속되어야 발화 시작으로 판단하고, 이후 2초 동안 기준 이하이면 발화 종료로 판단한다. 발화 길이에는 약 15초의 제한을 둔다.

`decode_audio()`는 입력 WAV 헤더가 16kHz 모노 PCM16인지 확인하고, 파형의 최대 진폭을 전수 조사해 최대 볼륨이 30 미만인 무음 입력이면 Whisper를 실행하지 않고 걸러낸다.

### 멀티모달 맥락 융합과 답변 생성 (`fuse_context` & `gemma_worker`)

`fuse_context()`는 Whisper가 변환한 음성 문자열, L 선택 여부, 문서 후보 지시 여부를 사전 결합한다.

| 인식 내용 | 처리 |
|---|---|
| L + “이걸로 떼줘” | 등본 안내로 해석 |
| 음성에 “등본” 또는 “주민등록” 포함 | L 없이도 등본 안내 가능 |
| 문서 후보만 있고 업무가 불명확함 | 업무를 다시 요청하는 안내 |
| 등본 신호와 타 업무(전입·복지·연금·인감·여권) 키워드 혼입 | 의도 충돌로 판정하여 확인 질문 출력 |

`Kiosk._process()`는 입력이 마감되는 즉시 비전 프로세스를 강제 종료(`_release_vision()`)한 뒤 Whisper(`whisper-cli`)로 음성을 문자로 변환하고, 비언어적 태그(`[음악]`, `(박수)` 등) 제거 및 300자 길이 제한 검사를 거쳐 `gemma_worker()`를 실행한다.

`gemma_worker()`는 4비트 양자화된 GGUF 모델을 불러와 음성 문자열(`voice`), 손동작(`gesture`), 업무(`service`), 문서 후보 지시 여부(`pointed_document_candidate`)를구조화된 JSON 문맥으로 묶어 프롬프트에 주입한다.
-프롬프트 복제 방지 및 환각 가드레일: 모델이 기본 안내문(`fallback`)을 그대로 베끼지 못하도록 프롬프트 입력 문맥(`context_for_llm`)에서는 `fallback` 필드를 분리해 주입한다. 또한 연결되지 않은 시스템의 '발급/인증 완료' 허위 주장을 정규표현식 필터로 감지해 차단하며, 프롬프트 인젝션 방어를 위해 입력 속 지시문 실행 금지 규칙을 명시한다.(시연 설정에 따라 사전 필터와 정규식 차단을 해제하여 100% 자유 생성 모드로도 구동할 수 있다.)
-의도적인 무상태(Stateless) 설계: 앞사람의 민원 대화 내역이 뒷사람의 프롬프트에 섞이는 개인정보 유출을 막고 LLM의 KV 캐시 메모리 증가를 방지하기 위해, 이전 턴의 대화 히스토리는 누적하지 않고 매 안내 종료 후 즉시 휘발시킨다.

### 음성 출력과 반이중(Half-Duplex) 오디오 제어

`tts_worker()`는 Piper 한국어 모델로 안내 문장을 합성하며, `Speaker.say()`가 이를 `aplay`로 재생한 뒤 임시 데이터를 즉시 삭제한다. 인사말이나 답변 음성이 스피커로 재생되는 동안에는 마이크 녹음 프로세스를 열지 않는반이중(Half-Duplex) 제어를 통해 키오스크가 자기 목소리를 다시 입력받는 에코 현상을 원천 차단했다.

### 화면과 실행 상태 관리 (안정성 장치)

`DesktopKiosk`는 카메라 영상, 인식 표시, 큰 안내 문장 및 운영 버튼을 담당한다.

-세대 카운터(`self.generation`)와 세션 토큰(`self.owner`): 빠른 운영 중지 후 재시작 시 이전 세션에서 뒤늦게 끝난 비동기 스레드 결과가 현재 화면을 덮어쓰는 레이스 컨디션(Race Condition)을 막기 위해, 세대 번호와 `secrets.token_hex` 소유권 토큰이 일치하는 이벤트만 화면에 반영한다.
-워치독(`_watchdog`) 자동 복구: 백그라운드 데몬 스레드가 1초 주기로 상태를 감시하여, 입력 캡처가 40초 이상 지연되거나 UI 응답이 45초 이상 끊기면 모든 하위 프로세스를 강제 정리하고 초기 대기 상태로 자동 복구한다.

| 주요 메서드 | 역할 |
|---|---|
| `start()` / `start_pump()` | 운영과 비전 프레임 수신 시작 (세대 카운터 갱신) |
| `frame()` | 영상과 인식 결과 표시, 접근·이탈 감지 및 `Hold` 판정 |
| `greet()` / `listen()` | 인사 완료 후 마이크 입력 시작 |
| `submit()` | 녹음을 종료하고 현재 멀티모달 입력 제출 |
| `present()` | 답변 표시 및 음성 안내 출력 |
| `abandon()` / `rearm()` | 입력 취소, 세션 초기화(Stateless) 및 비전 워커 재가동 |
| `tick()` / `event()` | 작업 결과 수신과 세대 번호 검증 후 화면 상태 갱신 |
| `stop()` / `cleanup()` | 카메라·마이크·재생·모델 워커 프로세스 일괄 종료 |

---

## 6. 메모리(OOM) 방어와 GPU 사용 전략

Jetson Orin Nano는 CPU와 GPU가 물리적 RAM을 공유하는 통합 메모리 구조이므로, 모든 AI 모델을 메모리에 상주시키면 즉각 OOM(Out of Memory)이 발생한다. 이를 해결하기 위해배타적 순차 실행(Exclusive Sequential Execution) 전략을 사용한다.

-비전 프로세스 강제 회수: 대기 및 입력 단계에서는 `vision_worker`(MediaPipe + YOLO)만 상주하며, 사용자의 입력이 마감되는 즉시 비전 프로세스를 `kill`하여 GPU/RAM 메모리를 OS로 100% 회수한다.
-순차 실행 및 즉시 종료: 확보된 메모리 공간 위에서 `whisper-cli` $\rightarrow$ `gemma_worker` $\rightarrow$ `tts_worker`를 하나씩 순서대로 실행하고, 각 작업이 끝나는 즉시 프로세스를 종료(`llm.close()`)하여 메모리 피크(Peak)치가 절대 겹치지 않게 통제한다.
-가중치 캐싱(`use_mmap=True`): 매 요청에서 모델을 새로 불러올 때 발생하는 Cold Start 지연을 완화하기 위해, Gemma 로드 시 메모리 맵핑(`mmap`)을 활성화하여 리눅스 페이지 캐시(Page Cache)에 올라간 모델 가중치를 디스크 재읽기 없이 재사용한다.
- Gemma는 기본 `--gpu-layers -1`로 GPU 오프로딩을 요청하며(`n_ctx=2048`, `n_batch=32`, `n_ubatch=32`), `--gpu-layers 0`으로 CPU 추론을 시험할 수도 있다. Piper는 CPU로 실행한다.

다른 무거운 프로세스(Jupyter 등)를 종료하고 다음 명령으로 실제 메모리 및 GPU 사용량을 실시간 확인할 수 있다.

```bash
tegrastats
```

---

## 7. 주요 옵션

| 옵션 | 용도 |
|---|---|
| `--fullscreen` | 전체 화면 시작 |
| `--autostart` | 창이 열린 뒤 자동 운영 시작 |
| `--no-llm` | Gemma 없이 입력 흐름 시험 |
| `--camera-device /dev/video2` | 카메라 장치 변경 |
| `--camera-backend gstreamer` | GStreamer 하드웨어 가속 파이프라인 방식 사용 |
| `--pipeline '…'` | 장치에 맞는 GStreamer 파이프라인 직접 지정 |
| `--person-detector yolo` | MediaPipe Pose 대신 YOLO로 사람 감지 |
| `--mic-device 이름` | ALSA 녹음 장치 선택 |
| `--speaker-device 이름` | ALSA 재생 장치 선택 |
| `--vad-threshold 0.015` | 음량 감지 기준, 기본값 `0.015` |
| `--gpu-layers 0` | Gemma CPU 추론 시험 (기본값 `-1`은 전체 GPU 오프로딩) |
| `--threads 2` | 작업 스레드 설정, 기본값 `2` |
| `--timeout 600` | Gemma 작업 제한 시간, 기본값 `600`초 |

`--vision-python`, `--llm-python`, `--tts-python`으로 각각의 독립된 Python 가상환경을 지정하여 패키지 버전 충돌을 우회할 수 있다. 모델 경로는 `--hand-model`, `--pose-model`, `--model`, `--tts-model`, `--whisper-bin`, `--whisper-model`, `--yolo-model`로 변경한다.

---

## 8. 문제 해결

### 기본 안내문만 계속 출력되는 경우

화면 오른쪽 아래(`result_mode` 영역)의 작은 상태 표시를 통해 원인을 바로 확인할 수 있다.
1.`기본 안내`라고 표시될 때: L 동작(0.7초 이상)이 인식되지 않았고 음성에도 '등본'이나 '주민등록' 키워드가 없거나, 타 업무 키워드(여권·인감 등)가 섞여 `fuse_context()` 단계에서 사전 차단된 경우이다.
2.`기본 안내 (완료 주장 차단)`이라고 표시될 때: Gemma가 정상적으로 답변을 생성했으나, 답변 내 '발급/인증' 및 '완료' 단어 조합이 정규표현식 가드레일(`fabricated`)에 걸려 기본 안내문으로 치환된 경우이다. 부정문("인증이 완료되지 않았습니다") 오판을 막도록 개선된 정규식을 적용하거나 가드레일을 완화하면 Gemma의 생성 문장이 그대로 출력된다.

### 소리가 나지 않는 경우

1. 프로그램은 운영 시작 직후가 아니라사람을 감지한 뒤 인사할 때 소리를 낸다.
2. 창 아래 Piper 또는 재생 실패 경고를 확인한다.
3. 운영을 중지하고 `--diagnose-audio`로 3초 녹음·재생을 확인한다.
4. Ubuntu 소리 설정에서 입력·출력 장치, 볼륨, 음소거를 확인한다. 모니터 자체에 스피커가 없는 경우 별도 스피커가 필요하다.
5. 녹음·재생은 되는데 안내만 안 나오면 `--check`에서 Piper Python(`.piper_venv`), ONNX 모델, `.onnx.json`을 확인한다.

### 카메라가 안 보이는 경우

`--diagnose-camera`로 프레임 수신을 확인한다. 장치 번호, 접근 권한, 다른 프로그램의 점유 여부를 확인한다. GStreamer를 사용한다면 OpenCV의 GStreamer 지원 여부와 카메라의 YUY2 $640 \times 480$ 30fps 지원 여부도 확인한다.

### 창이 열리지 않는 경우

`no display name` 또는 `$DISPLAY` 오류가 나면 Jetson 모니터의 Ubuntu 데스크톱 터미널에서 실행한다. Tkinter가 없으면 실행 Python에 맞는 시스템 Tk 패키지(`python3-tk`)가 필요하다.

### 말을 인식하지 못하거나 제출되지 않는 경우

`--list-audio`와 `--diagnose-audio`로 마이크를 확인한다. 주변 소음으로 2초 침묵 판정이 되지 않으면 환경을 조용하게 하거나 `--vad-threshold`를 조정한다. 입력 중 키보드의 `Enter` 또는 `Space` 키를 누르면 침묵 대기 없이 즉시 제출할 수 있다.

### 응답이 느리거나 메모리 오류가 나는 경우

Gemma 로딩과 추론 시간이 포함되어 있는지 화면의 처리 단계를 확인한다. `tegrastats`로 RAM 사용량을 확인하고 다른 프로세스를 종료한다. `--no-llm`으로 입력부만 먼저 점검한다. YOLO 엔진(`yolo11n_int8.engine`)은 현재 JetPack/TensorRT 환경과 호환되어야 한다.

---





