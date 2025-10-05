import cv2
import HandTrackingModule as htm
import numpy as np

# --- Utility: find a working camera index ---
def get_camera_index(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            return i
    return -1  # no camera found

# --- Initialize ---
camera_index = get_camera_index()
if camera_index == -1:
    raise Exception("❌ No webcam found. Please check camera permissions.")

cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
detector = htm.handDetector()

draw_color = (0, 0, 255)  # default red
img_canvas = np.zeros((720, 1280, 3), np.uint8)
xp, yp = 0, 0  # previous points for drawing

while True:
    success, frame = cap.read()
    if not success:
        print("⚠️ Failed to grab frame from webcam")
        break

    # Resize and flip
    frame = cv2.resize(frame, (1280, 720))
    frame = cv2.flip(frame, 1)

    # --- Draw 5 color selection boxes ---
    cv2.rectangle(frame, (10, 10), (230, 100), (0, 0, 255), cv2.FILLED)
    cv2.rectangle(frame, (240, 10), (460, 100), (0, 255, 0), cv2.FILLED)
    cv2.rectangle(frame, (470, 10), (690, 100), (255, 0, 0), cv2.FILLED)
    cv2.rectangle(frame, (700, 10), (920, 100), (0, 255, 255), cv2.FILLED)
    cv2.rectangle(frame, (940, 10), (1160, 100), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, "Eraser", (960, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # --- Hand detection ---
    frame = detector.findHands(frame, draw=True)
    lmList = detector.findPosition(frame, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]   # index finger
        x2, y2 = lmList[12][1:]  # middle finger

        fingers = detector.fingersUp()

        # --- Selection mode (two fingers up) ---
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0  # reset drawing point
            if y1 < 100:  # inside color selection bar
                if 10 < x1 < 230:
                    draw_color = (0, 0, 255)   # Red
                elif 240 < x1 < 460:
                    draw_color = (0, 255, 0)   # Green
                elif 470 < x1 < 690:
                    draw_color = (255, 0, 0)   # Blue
                elif 700 < x1 < 920:
                    draw_color = (0, 255, 255) # Yellow
                elif 940 < x1 < 1160:
                    draw_color = (0, 0, 0)     # Eraser

            cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, cv2.FILLED)

        # --- Drawing mode (index up only) ---
        if fingers[1] and not fingers[2]:
            cv2.putText(frame, 'Drawing mode', (900, 600),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
            cv2.circle(frame, (x1, y1), 15, draw_color, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if draw_color == (0, 0, 0):  # Eraser
                cv2.line(frame, (xp, yp), (x1, y1), draw_color, 50)
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, 50)
            else:
                cv2.line(frame, (xp, yp), (x1, y1), draw_color, 10)
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, 10)

            xp, yp = x1, y1

    # --- Merge drawings with live frame ---
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, img_inv)
    frame = cv2.bitwise_or(frame, img_canvas)

    cv2.imshow("Virtual Painter", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
