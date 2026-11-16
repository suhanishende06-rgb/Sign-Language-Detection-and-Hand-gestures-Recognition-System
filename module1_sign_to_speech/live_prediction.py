
import os
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mediapipe as mp

from collections import deque, Counter

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
TASK_MODEL = os.path.join(BASE_DIR, "hand_landmarker.task")

MODEL_PATH = os.path.join(MODEL_DIR, "isl_gcn_model.pth")
ADJACENCY_PATH = os.path.join(MODEL_DIR, "adjacency_matrix.npy")

# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device("cpu")

# --------------------------------------------------
# GCN MODEL
# --------------------------------------------------

class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency):
        x = torch.matmul(adjacency, x)
        return self.linear(x)


class ISL_GCN(nn.Module):

    def __init__(self, num_classes=26):
        super().__init__()

        self.gcn1 = GCNLayer(3, 32)
        self.gcn2 = GCNLayer(32, 64)
        self.gcn3 = GCNLayer(64, 64)

        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, adjacency, hand_mask):

        x = F.relu(self.gcn1(x, adjacency))
        x = F.relu(self.gcn2(x, adjacency))
        x = F.relu(self.gcn3(x, adjacency))

        node_mask = hand_mask.repeat_interleave(21, dim=1)
        node_mask = node_mask.unsqueeze(-1).float()

        pooled = (x * node_mask).sum(dim=1)
        pooled = pooled / node_mask.sum(dim=1).clamp(min=1)

        return self.classifier(pooled)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = ISL_GCN(num_classes=26).to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

adjacency = np.load(ADJACENCY_PATH)

adjacency = torch.tensor(
    adjacency,
    dtype=torch.float32
).to(device)

labels = list("abcdefghijklmnopqrstuvwxyz")

print("GCN Model Loaded Successfully!")


# --------------------------------------------------
# MEDIAPIPE HAND LANDMARKER
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK_MODEL),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)


# --------------------------------------------------
# HAND CONNECTIONS
# --------------------------------------------------

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

def extract_features(hand_results):

    all_hands = np.zeros((2, 21, 3), dtype=np.float32)
    hand_mask = np.zeros(2, dtype=np.float32)

    if not hand_results.hand_landmarks:
        return None, None

    detected_count = min(len(hand_results.hand_landmarks), 2)

    for hand_index in range(detected_count):

        hand_landmarks = hand_results.hand_landmarks[hand_index]

        points = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
            dtype=np.float32
        )

        wrist = points[0].copy()
        points = points - wrist

        scale = np.max(np.linalg.norm(points, axis=1))

        if scale > 1e-6:
            points = points / scale

        all_hands[hand_index] = points
        hand_mask[hand_index] = 1.0

    return all_hands, hand_mask


# --------------------------------------------------
# CAMERA START
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened!")
    landmarker.close()
    raise SystemExit

WINDOW_NAME = "ISL Sign Language Detection"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

history = deque(maxlen=20)

recognized_text = ""

current_letter = ""
current_confidence = 0.0

last_added_letter = ""
frame_timestamp = 0

print("\nCamera Started Successfully!")
print("Click on camera window before using keyboard controls.")


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Unable to read camera frame.")
            break

        frame = cv2.flip(frame, 1)

        frame_timestamp += 33

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        results = landmarker.detect_for_video(
            mp_image,
            frame_timestamp
        )

        features, hand_mask = extract_features(results)

        # ------------------------------------------
        # DRAW HAND LANDMARKS
        # ------------------------------------------

        if results.hand_landmarks:

            for hand_landmarks in results.hand_landmarks:

                points = []

                for landmark in hand_landmarks:

                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    points.append((x, y))

                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 255, 0),
                        -1
                    )

                for start, end in HAND_CONNECTIONS:

                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (255, 0, 0),
                        2
                    )

        # ------------------------------------------
        # MODEL PREDICTION
        # ------------------------------------------

        if features is not None:

            input_data = features.reshape(1, 42, 3)

            input_tensor = torch.tensor(
                input_data,
                dtype=torch.float32
            ).to(device)

            mask_tensor = torch.tensor(
                hand_mask.reshape(1, 2),
                dtype=torch.float32
            ).to(device)

            with torch.no_grad():

                output = model(
                    input_tensor,
                    adjacency,
                    mask_tensor
                )

                probabilities = torch.softmax(output, dim=1)

                confidence, predicted_index = torch.max(
                    probabilities,
                    dim=1
                )

            predicted_letter = labels[predicted_index.item()]
            prediction_confidence = confidence.item()

            history.append(predicted_letter)

            if len(history) == history.maxlen:

                counts = Counter(history)

                stable_letter, count = counts.most_common(1)[0]

                if count >= 14:

                    current_letter = stable_letter.upper()
                    current_confidence = prediction_confidence

                    if current_letter != last_added_letter:

                        recognized_text += current_letter
                        last_added_letter = current_letter

        else:

            history.clear()
            current_letter = ""
            current_confidence = 0.0
            last_added_letter = ""

        # ------------------------------------------
        # DISPLAY PREDICTION
        # ------------------------------------------

        cv2.putText(
            frame,
            "Indian Sign Language Detection",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Prediction: {current_letter if current_letter else 'Waiting...'}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {current_confidence * 100:.1f}%",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ------------------------------------------
        # RECOGNIZED TEXT
        # ------------------------------------------

        cv2.putText(
            frame,
            "Recognized Text:",
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        display_text = recognized_text[-45:]

        cv2.putText(
            frame,
            display_text,
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ------------------------------------------
        # BOTTOM KEYBOARD CONTROLS
        # SMALL WHITE TEXT
        # ------------------------------------------

        height, width = frame.shape[:2]

        controls = "Q - Quit    C - Clear Text    SPACE - Add Space"

        cv2.putText(
            frame,
            controls,
            (15, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # ------------------------------------------
        # SHOW CAMERA
        # ------------------------------------------

        cv2.imshow(WINDOW_NAME, frame)

        # ------------------------------------------
        # KEYBOARD CONTROLS
        # ------------------------------------------

        key = cv2.waitKeyEx(10)

        if key != -1:

            key = key & 0xFF

            if key == ord("q") or key == ord("Q"):

                print("Camera closing...")
                break

            elif key == ord("c") or key == ord("C"):

                recognized_text = ""
                history.clear()

                current_letter = ""
                current_confidence = 0.0
                last_added_letter = ""

                print("Text cleared!")

            elif key == 32:

                if recognized_text and not recognized_text.endswith(" "):
                    recognized_text += " "
                    last_added_letter = ""
                    history.clear()

                print("Space added!")

finally:

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

    print("Camera Closed Successfully!")
