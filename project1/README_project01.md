# Project01: 손 모양을 이용한 L·O·V·E 인식

## 1. 프로젝트 소개

NVIDIA Jetson Orin Nano에 연결한 USB 카메라에서 영상을 받아 손 모양을 L, O, V, E로 판별하는 프로그램이다. MediaPipe Hand Landmarker로 손의 관절 좌표를 추출하고, 관절 각도와 손 크기에 대한 거리 비율을 이용해 글자를 판별한다. 인식한 글자와 손 관절 연결선을 OpenCV 영상 창에 표시한다.

최대 두 손을 각각 판별하며, 글자를 특정 순서로 입력하거나 일정 시간 유지할 필요 없이 현재 프레임의 결과를 바로 표시한다. 여기서 사용하는 손 모양은 프로젝트에서 정한 제스처이며, 표준 수어 알파벳 전체를 인식하는 프로그램은 아니다.

## 2. 제출 파일 구성

프로젝트 파일은 GitHub 저장소의 `Project01` 폴더에 모은다. 수업 실습 원본과 구분하여 이 폴더 안의 `mp_test.py`를 프로젝트 실행 파일로 사용한다.

| 저장소 기준 경로 | 역할 |
| --- | --- |
| `Project01/README.md` | 프로젝트 설명, 실행 방법 및 의존성 안내 |
| `Project01/mp_love_project01.py` | 카메라 입력, 손 검출, 글자 판별, 결과 표시를 통합한 실행 파일 |
| `Project01/src/models/MediaPipe/hand_landmarker.task` | 손 검출에 필요한 MediaPipe 모델 파일 |

실행 코드는 `mp_test.py` 한 파일에 통합되어 있다. 별도의 `vision.py`, `jetson.py`, Flask 서버는 필요하지 않다. 모델 파일은 실행 코드와 별도로 필요하다.

## 3. 실행 환경 및 의존성

### 하드웨어 및 시스템

| 항목 | 필요한 조건 |
| --- | --- |
| 실행 장치 | NVIDIA Jetson Orin Nano |
| 운영 환경 | Jetson의 Linux 데스크톱과 Python 3 |
| 카메라 | `/dev/video0`으로 접근 가능한 USB 카메라 |
| 카메라 입력 형식 | YUY2, 640×480, 30fps를 요청하도록 설정 |
| 화면 출력 | OpenCV 영상 창을 표시할 수 있는 GUI 환경 |

카메라가 해당 입력 형식을 지원해야 한다. 입력 요청이 30fps라는 뜻이며, 손 인식까지 포함한 실제 처리 속도를 보장하는 수치는 아니다.

### 소프트웨어

| 의존성 | 용도 및 조건 |
| --- | --- |
| OpenCV (`cv2`) | GStreamer 카메라 입력, 색공간 변환, 선·원·텍스트 표시. GStreamer와 GUI 지원 빌드 필요 |
| NumPy (`numpy`) | 좌표 배열, 벡터 내적, 거리 및 각도 계산 |
| MediaPipe (`mediapipe`) | Tasks API의 `HandLandmarker`와 `HandLandmarksConnections` 사용 |
| GStreamer | `v4l2src`, `videoconvert`, `queue`, `appsink` 요소로 카메라 영상 전달 |
| `hand_landmarker.task` | MediaPipe 손 검출용 모델 |

수업에서 카메라와 MediaPipe 실행을 확인한 Jetson 파이썬 환경을 사용한다. Galaxy에서 사용한 가상환경이나 패키지 버전 목록을 Jetson에 그대로 복사해 설치하지 않는다. 특히 기존 OpenCV를 다른 패키지로 교체하면 필요한 GStreamer 또는 GUI 지원이 달라질 수 있다.

Jetson의 정확한 Python·패키지 버전은 실행 환경에서 아래 명령으로 확인한다. 확인하지 않은 버전 번호를 고정 설치 조건으로 지정하지 않았다.

```bash
python3 - <<'PY'
import sys
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

print("Python:", sys.version.split()[0])
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("MediaPipe:", mp.__version__)
print("HandLandmarker:", hasattr(vision, "HandLandmarker"))
print("HandLandmarksConnections:", hasattr(vision, "HandLandmarksConnections"))

for line in cv2.getBuildInformation().splitlines():
    if "GStreamer" in line or "GUI:" in line:
        print(line)
PY
```

`HandLandmarker`와 `HandLandmarksConnections`가 `True`, `GStreamer`가 `YES`인지 확인한다. `GUI`는 `NONE`이 아닌 화면 출력 지원 구성이 필요하다. 모듈 가져오기에서 오류가 나면 먼저 현재 활성화한 Python 환경과 설치된 패키지를 확인한다.

## 4. 실행 방법

1. USB 카메라를 Jetson에 연결한다.
2. `mp_love_prpject01.py`와 모델 파일을 위의 제출 폴더 구성대로 배치한다.
3. Jetson의 데스크톱 터미널에서 기존 실습용 Python 환경을 활성화한다. 별도 가상환경을 사용하지 않았다면 이 단계는 생략한다.
4. 저장소 최상위 폴더에서 아래 명령을 실행한다.

```bash
cd Project01
python3 mp_love_project01.py
```

모델 경로는 코드에 다음과 같이 지정되어 있다.

```python
MODEL_PATH = "src/models/MediaPipe/hand_landmarker.task"
```

이 상대 경로는 명령어를 실행한 현재 폴더 기준이므로 `Project01` 폴더 안에서 실행해야 한다. 모델이 없다면 수업에서 사용한 파일 또는 아래 참고 자료의 공식 Hand Landmarker 모델을 해당 위치에 배치한다.

영상 창 이름은 `MediaPipe Hand Detection - L O V E`이다. 영상 창을 선택한 상태에서 `q` 또는 `Esc`를 누르면 종료한다. 종료 시 카메라, 손 검출기와 OpenCV 창을 정리한다.

일반 SSH 연결만으로 실행하면 영상 창이 노트북에 자동으로 표시되지 않는다. 이 프로그램은 Jetson의 화면 출력이 가능한 데스크톱 세션에서 실행하는 구성을 기준으로 한다.

## 5. 사용 방법과 출력

카메라에 손 전체가 보이도록 놓고 아래 모양을 만든다. 두 손이 검출되면 각 손에 대해 별도의 글자를 표시한다.

| 글자 | 손 모양 |
| --- | --- |
| L | 엄지와 검지를 펴서 대략 직각을 만들고, 중지·약지·소지는 접는다. |
| O | 엄지와 검지 끝을 가까이 붙여 둥근 모양을 만들고, 나머지 세 손가락은 접는다. |
| V | 검지와 중지를 벌려 펴고, 약지·소지는 접으며 엄지는 바깥으로 펼치지 않는다. |
| E | 검지를 접고 중지·약지·소지를 펴며, 엄지는 바깥으로 펼치지 않는다. 프로젝트에서 정한 E 모양이다. |

| 화면 표시 | 의미 |
| --- | --- |
| `Hands: 0`, `Hands: 1`, `Hands: 2` | 현재 검출한 손 개수 |
| `Left: L / Right: V` 등 | 각 손의 라벨과 현재 판별한 글자. 출력 형태의 예시이며 실제 측정 결과를 뜻하지 않는다. |
| 손목 부근의 `L`, `O`, `V`, `E` | 해당 손에서 판별한 글자 |
| `-` | 손 모양이 판별 조건에 맞지 않음. 손이 없으면 상단에 `Letter: -` 표시 |
| 초록색 선 | 손 관절 연결선 |
| 빨간색 점 / 파란색 점 | 손가락 끝 / 그 외 관절 |

## 6. 처리 과정과 주요 함수

1. **영상 입력:** GStreamer 파이프라인으로 USB 카메라 프레임을 읽는다.
2. **이미지 조정:** 기존 실습 코드의 `cv2.flip(frame, 0)`으로 영상을 상하 반전한다.
3. **색공간 변환:** BGR 영상을 RGB로 변환해 MediaPipe 입력 이미지를 만든다.
4. **손 검출:** `hand_detector.detect()`로 최대 두 손의 관절 좌표를 얻는다. 현재 코드는 각 프레임에 기본 IMAGE 모드 검출을 적용한다.
5. **특징 계산:** 관절의 2D 픽셀 좌표에서 각도와 거리를 구한다. 손목 0번과 중지 뿌리 9번의 거리를 기준으로 거리를 정규화한다.
6. **글자 판별:** 손가락의 펴짐·접힘, 엄지와 검지의 거리, 손가락 사이 각도로 L·O·V·E를 판별한다.
7. **결과 표시:** 손 관절과 글자를 원본 프레임 위에 그려 OpenCV 창으로 출력한다.

| 함수 | 역할 |
| --- | --- |
| `angle(a, b, c)` | b를 중심으로 두 벡터가 이루는 각도를 도 단위로 계산 |
| `classify(p)` | 21개 관절 좌표로 손 모양 판별. 조건이 맞지 않으면 `None` 반환 |
| `draw_text(frame, text, position, color)` | 검정 테두리가 있는 글자를 영상에 표시 |
| `main()` | 모델과 카메라 준비, 프레임 반복 처리, 출력 및 자원 정리 |

MediaPipe 모델은 손 관절을 검출하고, L·O·V·E 글자는 직접 작성한 규칙으로 판별한다. 별도의 알파벳 분류 모델을 학습한 방식은 아니다.

### 주요 판별 기준

아래 거리는 모두 손목–중지 뿌리 거리를 1로 둔 상대 거리다.

| 항목 | 코드의 기준 |
| --- | --- |
| 손가락 펼침 | PIP 각도 > 150°, DIP 각도 > 145°, 손끝이 PIP보다 손목에서 멀리 있음 |
| 손가락 접힘 | 펼침 조건을 만족하지 않고, PIP < 150° 또는 DIP < 145° 또는 뿌리–손끝 거리 < 0.75 |
| 엄지 바깥 펼침 | 엄지 관절 각도 > 150°, 엄지 끝–검지 뿌리 거리 > 0.55 |
| O의 손끝 간격 | 엄지 끝–검지 끝 거리 < 0.35 |
| L의 벌어짐 | 엄지 방향과 검지 방향 사이 각도 45° 초과, 135° 미만 |
| V의 손끝 간격 | 검지 끝–중지 끝 거리 > 0.25 |

## 7. 참고 사항과 문제 해결

| 상황 | 확인할 내용 |
| --- | --- |
| 모델 파일 오류 | `Project01` 안에서 실행했는지와 `MODEL_PATH` 위치에 모델 파일이 있는지 확인한다. |
| 카메라 열기 실패 | `/dev/video0` 연결, 장치 사용 권한, 다른 프로그램의 점유, YUY2 640×480 30fps 지원, OpenCV의 GStreamer 지원을 확인한다. |
| 창이 표시되지 않음 | 실행 중인 세션의 GUI 사용 가능 여부와 OpenCV의 GUI 지원을 확인한다. |
| 영상이 거꾸로 표시됨 | 코드의 `frame = cv2.flip(frame, 0)`은 상하 반전이다. 카메라 방향에 반전이 필요 없다면 해당 줄을 제거한다. |
| 좌우 손 라벨이 반대로 표시됨 | 현재 코드는 기존 실습의 Left/Right 교환을 유지한다. 카메라 반전 방식에 따라 라벨 교환 부분을 확인한다. 이 라벨은 글자 판별 조건에는 사용하지 않는다. |
| 글자가 잘못 표시되거나 깜빡임 | 손 전체가 보이도록 하고 겹침·가림을 줄인다. 손 크기와 카메라 각도에 맞게 `classify()`의 각도·거리 기준을 조정한다. |

이 코드는 x, y 좌표만 사용하므로 손을 옆으로 크게 돌리거나 손가락이 가려지면 판별이 어려울 수 있다. 프레임별 결과를 즉시 표시하므로 손 모양 전환 중에는 글자가 바뀌거나 `-`가 나타날 수 있다.

현재 구현 범위는 이미지 반전, 색공간 변환, 손 관절 검출, 기하학적 특징 계산 및 결과 overlay이다. 별도의 영상 명도 강화, 색상 분할, 컨투어 검출, 칼만 필터 기반 추적은 포함하지 않는다. 과제에서 해당 기능을 모두 요구하는 경우에는 추가 구현이 필요하다.

문법 검사와 가상 관절 좌표를 이용한 L·O·V·E 판별 및 크기·위치·반전 변환 검사는 통과했다. 실제 Jetson 카메라의 통합 실행, 인식률과 처리 FPS는 별도로 측정해야 하며, 이 문서에는 실측 성능을 기재하지 않았다.

## 8. GitHub 제출

`Project01` 폴더에 실행 코드, 이 README와 필요한 모델 파일을 포함해 본인 GitHub 저장소에 제출한다. `.venv`와 `__pycache__` 같은 실행 환경 및 임시 파일은 제출 코드에 포함하지 않는다.

저장소 최상위 폴더에서 변경 파일을 확인한 뒤 다음과 같이 제출할 수 있다.

```bash
git status
git add Project01/README.md Project01/mp_test.py Project01/src/models/MediaPipe/hand_landmarker.task
git commit -m "feat: Jetson 손 관절 기반 L O V E 인식 및 실행 안내 추가"
git push
```

제출 시각은 해당 파일이 포함된 커밋의 기록 시각을 기준으로 한다. 커밋 메시지는 변경 내용을 구체적으로 설명한다. 최종 프로젝트 제출 단계에서는 수업 지침에 따라 발표 자료 PDF도 같은 프로젝트 폴더에 추가한다.

## 9. 참고 자료

- [MediaPipe Hand Landmarker Python 공식 안내](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)
- [MediaPipe Hand Landmarker 개요 및 모델 안내](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker#models)

손 인식 모델과 API 사용법은 위 공식 자료를 참고하며, 글자별 손 모양과 판별 규칙은 이 프로젝트에서 정의했다.
