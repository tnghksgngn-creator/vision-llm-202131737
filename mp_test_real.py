import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# 기존 프로젝트 폴더에서 실행: python3 mp_test.py
MODEL_PATH = "src/models/MediaPipe/hand_landmarker.task"

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

def angle(a, b, c):
    """b 관절의 각도: 일직선이면 약 180도."""
    u, v = a - b, c - b
    denominator = np.linalg.norm(u) * np.linalg.norm(v)

    if denominator < 1e-8:
        return 0.0

    cosine = np.clip(np.dot(u, v) / denominator, -1, 1)
    return float(np.degrees(np.arccos(cosine)))


def classify(p):
    """손 관절 21개의 픽셀 좌표로 L/O/V/E를 판별."""
    scale = np.linalg.norm(p[0] - p[9])
    if scale < 1:
        return None

    def distance(i, j):
        return np.linalg.norm(p[i] - p[j]) / scale

    # 검지, 중지, 약지, 소지: 뿌리 / 중간 관절 / 끝 관절 / 손끝
    fingers = [
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20)
    ]

    extended = []
    folded = []

    for base, pip, dip, tip in fingers:
        pip_angle = angle(p[base], p[pip], p[dip])
        dip_angle = angle(p[pip], p[dip], p[tip])

        is_extended = (
            pip_angle > 150
            and dip_angle > 145
            and distance(0, tip) > distance(0, pip)
        )
        extended.append(is_extended)

        folded.append(
            not is_extended
            and (
                pip_angle < 150
                or dip_angle < 145
                or distance(base, tip) < 0.75
            )
        )

    index, middle, ring, pinky = extended
    thumb_straight = angle(p[2], p[3], p[4]) > 150
    thumb_out = thumb_straight and distance(4, 5) > 0.55

    # 사용자 지정 E: 검지를 접고 중지/약지/소지는 펼치기
    if folded[0] and middle and ring and pinky and not thumb_out:
        return "E"

    # O: 엄지와 검지 끝을 붙이고 나머지 세 손가락 접기
    if distance(4, 8) < 0.35 and not index and all(folded[1:]):
        return "O"

    # L: 엄지와 검지를 펴서 대략 직각 만들기
    thumb_direction = p[4] - p[2]
    index_direction = p[8] - p[5]
    spread_angle = angle(thumb_direction, np.zeros(2), index_direction)

    if (
        index
        and all(folded[1:])
        and thumb_out
        and 45 < spread_angle < 135
    ):
        return "L"

    # V: 검지와 중지를 벌려 펴고 약지/소지는 접기
    if (
        index and middle
        and folded[2] and folded[3]
        and distance(8, 12) > 0.25
        and not thumb_out
    ):
        return "V"

    return None


def draw_text(frame, text, position, color=(0, 255, 255)):
    """검정 테두리를 넣어 배경 위에서도 글자가 잘 보이게 표시."""
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4
    )
    cv2.putText(
        frame, text, position,
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
    )


def main():
    # 기존 코드의 detect() 방식을 사용하므로 타임스탬프는 필요 없음.
    base_option = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_option,
        num_hands=2
    )
    connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
    finger_tips = (4, 8, 12, 16, 20)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    hand_detector = None

    try:
        if not cap.isOpened():
            raise RuntimeError(
                "카메라를 열 수 없습니다. /dev/video0와 GStreamer 설정을 확인하세요."
            )

        hand_detector = vision.HandLandmarker.create_from_options(options)
        print("실행 중: 영상 창에서 q 또는 ESC를 누르면 종료합니다.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("카메라 프레임을 읽지 못했습니다.")
                break

            # 기존 카메라 방향 유지: 0은 상하 반전, 1은 좌우 반전.
            frame = cv2.flip(frame, 0)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 선이나 글자를 그리기 전 영상으로 손 탐지
            result = hand_detector.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb)
                )
            )

            h, w = frame.shape[:2]
            # 기존 실습 코드의 좌우 라벨 변환 유지
            labels = [
                "Left" if hand[0].category_name == "Right" else "Right"
                for hand in result.handedness
            ]
            summaries = []

            for hand_index, hand in enumerate(result.hand_landmarks):
                # 판별은 실수 좌표, 그림은 정수 좌표 사용
                p = np.array(
                    [[landmark.x * w, landmark.y * h] for landmark in hand],
                    dtype=np.float32
                )
                letter = classify(p)
                points = [tuple(map(int, point)) for point in p]
                label = labels[hand_index] if hand_index < len(labels) else "Hand"
                summaries.append(f"{label}: {letter or '-'}")

                # 손 관절 연결선
                for connection in connections:
                    cv2.line(
                        frame, points[connection.start], points[connection.end],
                        (0, 255, 0), 2
                    )

                # 손끝은 빨간 점, 나머지 관절은 파란 점
                for i, point in enumerate(points):
                    color = (0, 0, 255) if i in finger_tips else (255, 0, 0)
                    radius = 6 if i in finger_tips else 4
                    cv2.circle(frame, point, radius, color, -1)

                # 각 손의 손목 근처에 현재 글자만 즉시 표시
                position = (
                    max(10, min(w - 40, points[0][0])),
                    max(100, min(h - 10, points[0][1] + 30))
                )
                color = (0, 255, 0) if letter is not None else (255, 255, 255)
                draw_text(frame, letter or "-", position, color)

            draw_text(frame, f"Hands: {len(result.hand_landmarks)}", (20, 35))
            draw_text(frame, " / ".join(summaries) or "Letter: -", (20, 70))

            cv2.imshow("MediaPipe Hand Detection - L O V E", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        cap.release()
        try:
            if hand_detector is not None:
                hand_detector.close()
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
