# Face Detection System using DeepFace and OpenCV

Real-time face detection and recognition in Python with multi-camera support.

## Features
- Live face detection (OpenCV Haar cascade)
- Face recognition against a folder of known people (DeepFace, VGG-Face)
- One thread per camera so multiple cameras run smoothly
- Recognition runs every N frames to keep video responsive

## Setup
```bash
pip install -r requirements.txt
```
Add one clear photo per person to `known_faces/` (file name = person's name).

## Run
```bash
python face_system.py --cameras 0        # single webcam
python face_system.py --cameras 0 1      # two cameras
```
Press `q` to quit.

## Tech stack
Python, OpenCV, DeepFace, NumPy
