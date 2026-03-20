import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.spatial import distance
import time
import os

MODEL_PATH = "models/face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

def collect_ear_samples(cap, label, duration=5):
    print(f"\n>>> {label} — collecting for {duration} seconds...")
    samples = []
    start = time.time()
    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results  = face_landmarker.detect(mp_image)

        remaining = duration - (time.time() - start)

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            left_ear  = eye_aspect_ratio(landmarks, LEFT_EYE,  w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            avg_ear   = (left_ear + right_ear) / 2.0
            samples.append(avg_ear)
            cv2.putText(frame, f"{label}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"EAR: {avg_ear:.3f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Time left: {remaining:.1f}s", (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return samples

print("=" * 50)
print("   EAR CALIBRATION TOOL")
print("=" * 50)
print("This will measure your personal EAR values")
print("Keep your face in front of the camera")
print("Press any key to start each step")
print("=" * 50)

cap = cv2.VideoCapture(0)

# Step 1 — open eyes
input("\nStep 1: OPEN your eyes wide and press Enter...")
open_samples = collect_ear_samples(cap, "EYES OPEN - look at camera", duration=5)

# Step 2 — closed eyes  
input("\nStep 2: CLOSE your eyes completely and press Enter...")
closed_samples = collect_ear_samples(cap, "EYES CLOSED", duration=5)

cap.release()
cv2.destroyAllWindows()

if not open_samples or not closed_samples:
    print("Not enough samples collected, try again!")
else:
    open_mean   = np.mean(open_samples)
    closed_mean = np.mean(closed_samples)
    threshold   = (open_mean + closed_mean) / 2.0

    print("\n" + "=" * 50)
    print("   CALIBRATION RESULTS")
    print("=" * 50)
    print(f"Eyes OPEN  average EAR : {open_mean:.3f}")
    print(f"Eyes CLOSED average EAR: {closed_mean:.3f}")
    print(f"Recommended threshold  : {threshold:.3f}")
    print("=" * 50)

    with open("models/ear_threshold.txt", "w") as f:
        f.write(str(round(threshold, 3)))
    print(f"Threshold saved to models/ear_threshold.txt")
    print("Restart your backend to apply!")