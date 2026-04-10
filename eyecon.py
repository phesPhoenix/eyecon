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

# MediaPipe face mesh landmark indices for each eye
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Tuning parameters
WINK_THRESHOLD     = 0.15  # EAR below this = eye closed
OPEN_THRESHOLD     = 0.18  # EAR above this = eye open
WINK_FRAMES_REQ    = 8     # frames eye must stay closed to register a wink
WINK_COOLDOWN      = 0.8   # seconds between clicks

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """Compute Eye Aspect Ratio (EAR) for blink/wink detection."""
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    v1 = abs(points[1][1] - points[5][1])
    v2 = abs(points[2][1] - points[4][1])
    h1 = abs(points[0][0] - points[3][0])
    return (v1 + v2) / (2.0 * h1), points

# Download MediaPipe face landmarker model if not present
MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading face landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MODEL_PATH
    )

# Initialize face landmarker
options  = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=MODEL_PATH), num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# State
cap               = cv2.VideoCapture(0)
left_wink_frames  = 0
right_wink_frames = 0
last_wink_time    = 0
frame_count       = 0
results           = None
left_ear_history  = deque(maxlen=3)
right_ear_history = deque(maxlen=3)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame_count += 1

    # Run detection every 3 frames for performance
    if frame_count % 3 == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    if results and results.face_landmarks:
        landmarks = results.face_landmarks[0]

        left_ear,  left_pts  = eye_aspect_ratio(landmarks, LEFT_EYE,  w, h)
        right_ear, right_pts = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)

        # Smooth EAR over last 3 frames
        left_ear_history.append(left_ear)
        right_ear_history.append(right_ear)
        left_ear  = sum(left_ear_history)  / len(left_ear_history)
        right_ear = sum(right_ear_history) / len(right_ear_history)

        # Draw eye landmarks
        for pt in left_pts + right_pts:
            cv2.circle(frame, pt, 3, (0, 255, 0), -1)

        cv2.putText(frame, f"Left EAR: {left_ear:.2f}",  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        now = time.time()

        # Left wink -> left click
        if left_ear < WINK_THRESHOLD and right_ear > OPEN_THRESHOLD:
            left_wink_frames += 1
            if left_wink_frames >= WINK_FRAMES_REQ and now - last_wink_time > WINK_COOLDOWN:
                cv2.putText(frame, "LEFT WINK - LEFT CLICK", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                mouse.click(Button.left, 1)
                last_wink_time = now
                left_wink_frames = 0
        else:
            left_wink_frames = 0

        # Right wink -> right click
        if right_ear < WINK_THRESHOLD and left_ear > OPEN_THRESHOLD:
            right_wink_frames += 1
            if right_wink_frames >= WINK_FRAMES_REQ and now - last_wink_time > WINK_COOLDOWN:
                cv2.putText(frame, "RIGHT WINK - RIGHT CLICK", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                mouse.click(Button.right, 1)
                last_wink_time = now
                right_wink_frames = 0
        else:
            right_wink_frames = 0

    cv2.imshow("EyeCon", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
