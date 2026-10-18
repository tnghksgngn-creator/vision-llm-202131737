import cv2
import numpy as np
pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)


green_lower = np.array([30, 60, 90], dtype=np.uint8)
green_upper = np.array([230, 115, 180], dtype=np.uint8)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_blur = cv2.GaussianBlur(frame,(7,7),0)
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    mask1 = cv2.inRange(frame_lab, green_lower, green_upper)
    mask2 = cv2.inRange(frame_lab, green_lower, green_upper)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
    mask,        # binary mask
    cv2.RETR_EXTERNAL,       # 구멍 뚫린 객체의 바깥쪽 테두리만 검출
    cv2.CHAIN_APPROX_SIMPLE  # 직선 경로 위에 있는 픽셀 좌표 전체 대신 꼭짓점 좌표만 저장
    )

    if contours:
        contour= max(contours,key=cv2.contourArea)

        if cv2.contourArea(contour) >100:
            (x,y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x),int(y))

            cv2.circle(frame,center, int(radius),(0,255,255),2)
            cv2.circle(frame,center,4,(0,0,255),-1)
    cv2.imshow("green ball",frame)
    cv2.imshow("mask",mask)

    cv2.imshow("VideoCapture with GStreamer", frame)


cap.release()
cv2.destroyAllWindows()