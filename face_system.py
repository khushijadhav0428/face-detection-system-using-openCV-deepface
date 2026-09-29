"""Real-time face recognition with DeepFace + OpenCV (multi-camera support).

Usage:
    python face_system.py --cameras 0            # one webcam
    python face_system.py --cameras 0 1          # two cameras
Put one photo per person in known_faces/ (file name = person's name, e.g. khushi.jpg).
Press 'q' in any window to quit.
"""
import argparse
import os
import threading
import time

import cv2
from deepface import DeepFace

KNOWN_DIR = "known_faces"
MODEL = "VGG-Face"
DETECTOR = "opencv"
ANALYZE_EVERY_N_FRAMES = 15  # recognition is heavy, so don't run it every frame


def load_known_faces():
    """Return {name: image_path} for every image in known_faces/."""
    known = {}
    for fname in os.listdir(KNOWN_DIR):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            known[os.path.splitext(fname)[0]] = os.path.join(KNOWN_DIR, fname)
    return known


def identify(frame, known):
    """Compare the face in `frame` with each known face. Returns best name or 'Unknown'."""
    best_name, best_dist = "Unknown", 1.0
    for name, path in known.items():
        try:
            res = DeepFace.verify(
                frame, path, model_name=MODEL,
                detector_backend=DETECTOR, enforce_detection=True,
            )
        except ValueError:  # no face found in frame
            return "No face"
        if res["verified"] and res["distance"] < best_dist:
            best_name, best_dist = name, res["distance"]
    return best_name


class CameraWorker(threading.Thread):
    """One thread per camera so a slow camera doesn't block the others."""

    def __init__(self, cam_id, known):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.known = known
        self.cap = cv2.VideoCapture(cam_id)
        self.label = "..."
        self.running = True
        self.frame = None
        self.faces = []

    def run(self):
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        count = 0
        while self.running and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.faces = face_cascade.detectMultiScale(gray, 1.2, 5)
            if count % ANALYZE_EVERY_N_FRAMES == 0 and len(self.faces):
                self.label = identify(frame, self.known)
            self.frame = frame
            count += 1
        self.cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    known = load_known_faces()
    if not known:
        print("Add at least one image to known_faces/ first.")
        return

    workers = [CameraWorker(c, known) for c in args.cameras]
    for w in workers:
        w.start()

    while True:
        for w in workers:
            if w.frame is None:
                continue
            view = w.frame.copy()
            for (x, y, fw, fh) in w.faces:
                cv2.rectangle(view, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
                cv2.putText(view, w.label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(f"Camera {w.cam_id}", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        time.sleep(0.01)

    for w in workers:
        w.running = False
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
