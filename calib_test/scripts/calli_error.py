import cv2
import numpy as np

# 1. 흰 배경 이미지 생성 (크기: 1080x1920, 3채널 BGR)
height, width = 1440, 1080
img = np.ones((height, width, 3), dtype=np.uint8) * 255  # 흰색

# 2. 예시 픽셀 좌표 (u, v)
u, v = 828.83, 726.66
a,b = 829.329, 727.677
c,d = 837.05, 690.46
e,f = 836.56, 689.92

# 3. 이미지 위에 빨간 점 찍기 (반지름 5)
cv2.circle(img, (int(u), int(v)), 2, (0, 0, 255), -1)
cv2.circle(img, (int(a), int(b)), 2, (0, 255, 0), -1)
cv2.circle(img, (int(c), int(d)), 2, (255, 0, 0), -1)
cv2.circle(img, (int(e), int(f)), 2, (255, 255, 0), -1)
# 4. 창에 출력
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
