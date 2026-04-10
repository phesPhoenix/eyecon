# EyeCon

Hands-free mouse input using eye tracking. Wink your left eye to left click, wink your right eye to right click — no hands required.

## How it works

EyeCon uses MediaPipe's face landmarker to track 6 landmarks around each eye and computes the **Eye Aspect Ratio (EAR)** — a ratio of vertical to horizontal eye openness. When one eye's EAR drops below a threshold while the other stays open, it registers as a wink and fires a mouse click via `pynput`.

To reduce false positives, a wink must be held for 8 consecutive frames and a cooldown prevents rapid re-triggering.

## Stack

- Python, OpenCV, MediaPipe, pynput

## Setup

```bash
pip install opencv-python mediapipe pynput
python eyecon.py
```

The face landmarker model downloads automatically on first run.

## Controls

| Action | Input |
|--------|-------|
| Left click | Left wink |
| Right click | Right wink |
| Quit | ESC |
