import cv2

cap = cv2.VideoCapture(1, cv2.CAP_V4L2)  # CAP_V4L2 명시

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
if not cap.isOpened():
    print("❌ 카메라 열기 실패")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 수신 실패")
        continue

    cv2.imshow("Cam", frame)
    if cv2.waitKey(1) == 27:  # ESC 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
