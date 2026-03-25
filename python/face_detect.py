import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request
import time
from pynput.mouse import Button, Controller
from collections import deque

mouse = Controller()

# Eye landmark indices for MediaPipe face mesh
# Left eye
LEFT_EYE = [362, 385, 387, 263, 373, 380]
# Right eye
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    points = []
    for i in eye_indices:
        x = int(landmarks[i].x * w)
        y = int(landmarks[i].y * h)
        points.append((x, y))

    # vertical distances
    v1 = abs(points[1][1] - points[5][1])
    v2 = abs(points[2][1] - points[4][1])
    # horizontal distance
    h1 = abs(points[0][0] - points[3][0])

    ear = (v1 + v2) / (2.0 * h1)
    return ear, points

model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        model_path
    )

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
left_wink_frames = 0
right_wink_frames = 0
last_wink_time = 0
WINK_THRESHOLD = 0.15
OPEN_THRESHOLD = 0.18
WINK_FRAMES_REQUIRED = 8
WINK_COOLDOWN = 0.8
frame_count = 0
results = None
left_ear_history = deque(maxlen=3)
right_ear_history = deque(maxlen=3)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    process_this_frame = (frame_count % 3 == 0)

    h, w = frame.shape[:2]

    if process_this_frame:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)

    if results and results.face_landmarks:
        landmarks = results.face_landmarks[0]

        left_ear, left_points = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear, right_points = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)

        left_ear_history.append(left_ear)
        right_ear_history.append(right_ear)
        left_ear = sum(left_ear_history) / len(left_ear_history)
        right_ear = sum(right_ear_history) / len(right_ear_history)

        # draw eye points
        for point in left_points:
            cv2.circle(frame, point, 3, (0, 255, 0), -1)
        for point in right_points:
            cv2.circle(frame, point, 3, (0, 255, 0), -1)

        # display EAR values
        cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # left wink
        if left_ear < WINK_THRESHOLD and right_ear > OPEN_THRESHOLD:
            left_wink_frames += 1
            if left_wink_frames >= WINK_FRAMES_REQUIRED:
                if time.time() - last_wink_time > WINK_COOLDOWN:
                    cv2.putText(frame, "LEFT WINK - LEFT CLICK", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    print("LEFT CLICK")
                    mouse.click(Button.left, 1)
                    last_wink_time = time.time()
                    left_wink_frames = 0
        else:
            left_wink_frames = 0

        # right wink
        if right_ear < WINK_THRESHOLD and left_ear > OPEN_THRESHOLD:
            right_wink_frames += 1
            if right_wink_frames >= WINK_FRAMES_REQUIRED:
                if time.time() - last_wink_time > WINK_COOLDOWN:
                    cv2.putText(frame, "RIGHT WINK - RIGHT CLICK", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    print("RIGHT CLICK")
                    mouse.click(Button.right, 1)
                    last_wink_time = time.time()
                    right_wink_frames = 0
        else:
            right_wink_frames = 0

    cv2.imshow("EyeCon - EAR Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()