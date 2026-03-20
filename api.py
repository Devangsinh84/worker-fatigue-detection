import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from scipy.spatial import distance
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import uvicorn
import urllib.request
import os

EAR_THRESHOLD = 0.204
try:
    with open("models/ear_threshold.txt", "r") as f:
        EAR_THRESHOLD = float(f.read().strip())
    print(f"Loaded EAR threshold: {EAR_THRESHOLD}")
except:
    print(f"Using default EAR threshold: {EAR_THRESHOLD}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("models/eye_state_model.keras")

MODEL_PATH = "models/face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading face landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MODEL_PATH
    )
    print("Downloaded!")

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

def extract_eye(frame, landmarks, eye_indices, w, h, size=97):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    x_coords = [p[0] for p in pts]
    y_coords = [p[1] for p in pts]
    x1, x2 = max(min(x_coords) - 10, 0), min(max(x_coords) + 10, w)
    y1, y2 = max(min(y_coords) - 10, 0), min(max(y_coords) + 10, h)
    eye = frame[y1:y2, x1:x2]
    if eye.size == 0:
        return None
    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    eye = cv2.resize(eye, (size, size))
    return eye.reshape(1, size, size, 1).astype("float32") / 255.0

@app.get("/")
def root():
    return {"status": "Worker Fatigue Detection API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image"}

    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )
    results = face_landmarker.detect(mp_image)

    if not results.face_landmarks:
        return {
            "face_detected": False,
            "eyes_closed":   False,
            "fatigue":       False,
            "ear":           float(0.0),
            "cnn_score":     float(0.0)
        }

    landmarks = results.face_landmarks[0]
    left_ear  = eye_aspect_ratio(landmarks, LEFT_EYE,  w, h)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
    avg_ear   = round((left_ear + right_ear) / 2.0, 3)

    # Pure EAR based detection using your calibrated threshold
    eyes_closed = bool(avg_ear < EAR_THRESHOLD)

    # Get face bounding box from landmarks
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    
    x_min = float(min(x_coords))
    x_max = float(max(x_coords))
    y_min = float(min(y_coords))
    y_max = float(max(y_coords))

    return {
        "face_detected": True,
        "eyes_closed":   bool(eyes_closed),
        "fatigue":       bool(eyes_closed),
        "ear":           float(avg_ear),
        "cnn_score":     float(0.0),
        "face_box": {
            "x":      x_min,
            "y":      y_min,
            "width":  x_max - x_min,
            "height": y_max - y_min
        }
    }
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)