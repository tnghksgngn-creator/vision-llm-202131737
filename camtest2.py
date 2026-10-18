import cv2
import numpy as np
from numpy.linalg import inv

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
def KalmanFilter(mu_prev, sigma_prev, z):
    mu_bar = A_t.dot(mu_prev)
    sigma_bar = A_t.dot(sigma_prev).dot(A_t.transpose()) + R_t
    if z is None:
        return mu_bar, sigma_bar
    else:
        K_t = sigma_bar.dot(C_t.transpose()).dot(inv(C_t.dot(sigma_bar).dot(C_t.transpose()) + Q_t))
        mu = mu_bar + K_t.dot(z - C_t.dot(mu_bar))
        sigma = (np.identity(2) - K_t.dot(C_t)).dot(sigma_bar)
        return mu, sigma


# Kalman filter 변수 정의
A_t = np.array([[1, 1], [0, 1]])
G = np.array([[0.5], [1]])
R_t = G.dot(G.transpose())
C_t = np.array([[1, 0]])
Q_t = np.array([[1]])
mu_t = np.array([[0, 0], [0, 0]])
sigma_t = np.array([[0, 0], [0, 0]])

found=False
radius = 10


while True:
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_blur = cv2.GaussianBlur(frame,(7,7),0)
    frame_lab = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2LAB)

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

    if len(contours) > 0:
        # 가장 큰 contour 선택
        contour= max(contours,key=cv2.contourArea)
        if cv2.contourArea(contour) >100:
                (x,y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x),int(y))
                z=np.array([[x,y]],dtype=float)
        
                cv2.circle(frame,center, int(radius),(0,255,255),2)
                cv2.circle(frame,center,4,(0,0,255),-1)

        # 검출된 객체에 파란 원 overlay
        # cv2.circle(frame, center, int(radius), ...)
        # cv2.circle(frame, center, ...)

        # 객체 최초 검출
        if not found:
            found = True
    
    
    
    if found and (len(contours) > 0):
        mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, np.array([list(center)]))
        x_bel, y_bel = mu_t[0][0], mu_t[0][1]

    # 측정값 미사용 (occluded) -> Prediction
    elif found and (len(contours) <= 0):
        mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, None)
        estimated_center = (int(mu_t[0,0]),int(mu_t[0,1]))

        x_bel, y_bel = mu_t[0][0], mu_t[0][1]
        cv2.circle(frame,estimated_center, int(radius)+3,(255,0,0),2)
        cv2.circle(frame,estimated_center,4,(0,255,255))    

    cv2.imshow("green ball",frame)
    cv2.imshow("mask",mask)

    cv2.imshow("VideoCapture with GStreamer", frame)


cap.release()
cv2.destroyAllWindows()