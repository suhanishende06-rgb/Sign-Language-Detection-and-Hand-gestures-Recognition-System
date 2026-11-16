
import os
import json
import re
import time
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
from collections import deque, Counter

from sentence_builder import SentenceBuilder
from speech_output import SpeechOutput

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASK_MODEL = os.path.join(BASE_DIR, "hand_landmarker.task")
INCLUDE_MODEL_PATH = os.path.join(BASE_DIR, "models", "isl_word_bilstm_include.pth")
INCLUDE_LABEL_PATH = os.path.join(BASE_DIR, "models", "word_labels_include.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "isl_word_bilstm.pth")
LABEL_PATH = os.path.join(BASE_DIR, "models", "word_labels.json")
STATIC_MODEL_PATH = os.path.join(BASE_DIR, "models", "isl_gcn_model.pth")
ADJACENCY_PATH = os.path.join(BASE_DIR, "models", "adjacency_matrix.npy")

if os.path.exists(INCLUDE_MODEL_PATH) != os.path.exists(INCLUDE_LABEL_PATH):
    raise FileNotFoundError(
        "The INCLUDE model and its labels must both exist. "
        "Run train_include_word_model.py to finish training."
    )
if os.path.exists(INCLUDE_MODEL_PATH):
    MODEL_PATH = INCLUDE_MODEL_PATH
    LABEL_PATH = INCLUDE_LABEL_PATH

SEQUENCE_LENGTH = 30
MODEL_SEQUENCE_LENGTH = 30
FEATURE_COUNT = 126
DYNAMIC_MOTION_THRESHOLD = 0.0035
HAND_LOSS_RESET_SECONDS = 0.5
INFERENCE_INTERVAL = 3
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

# --------------------------------------------------
# CHECK FILES
# --------------------------------------------------

for path in [
    TASK_MODEL,
    MODEL_PATH,
    LABEL_PATH,
    STATIC_MODEL_PATH,
    ADJACENCY_PATH,
]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

# --------------------------------------------------
# DEVICE AND LABELS
# --------------------------------------------------

device = torch.device("cpu")
torch.set_num_threads(1)

with open(LABEL_PATH, "r", encoding="utf-8") as label_file:
    classes = json.load(label_file)

print("Loaded classes:", classes)

# --------------------------------------------------
# BiLSTM MODEL
# Same architecture as training script
# --------------------------------------------------

class WordBiLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=126,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        output, _ = self.lstm(x)
        features = output[:, -1, :]
        return self.fc(self.dropout(features))


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency):
        return self.linear(torch.matmul(adjacency, x))


class StaticSignGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = GCNLayer(3, 32)
        self.gcn2 = GCNLayer(32, 64)
        self.gcn3 = GCNLayer(64, 64)
        self.classifier = nn.Linear(64, 26)

    def forward(self, x, adjacency, hand_mask):
        x = torch.relu(self.gcn1(x, adjacency))
        x = torch.relu(self.gcn2(x, adjacency))
        x = torch.relu(self.gcn3(x, adjacency))
        node_mask = hand_mask.repeat_interleave(21, dim=1).unsqueeze(-1).float()
        pooled = (x * node_mask).sum(dim=1)
        pooled = pooled / node_mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


model = WordBiLSTM(len(classes)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("Word model loaded successfully!")

static_model = StaticSignGCN().to(device)
static_model.load_state_dict(
    torch.load(STATIC_MODEL_PATH, map_location=device, weights_only=True)
)
static_model.eval()
adjacency = torch.tensor(
    np.load(ADJACENCY_PATH),
    dtype=torch.float32,
    device=device,
)
static_labels = list("abcdefghijklmnopqrstuvwxyz")

print("Static letter model loaded successfully!")

# --------------------------------------------------
# MEDIAPIPE
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK_MODEL),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.35,
    min_hand_presence_confidence=0.35,
    min_tracking_confidence=0.35
)

landmarker = HandLandmarker.create_from_options(options)

# --------------------------------------------------
# FEATURE EXTRACTION
# Exactly matching sequence_collector.py
# --------------------------------------------------

def extract_features(results):

    features = np.zeros((2, 21, 3), dtype=np.float32)

    if not results.hand_landmarks:
        return None

    for index, hand_landmarks in enumerate(results.hand_landmarks[:2]):

        slot = index

        if (
            results.handedness
            and index < len(results.handedness)
            and results.handedness[index]
        ):
            hand_name = results.handedness[index][0].category_name

            if hand_name.lower() == "left":
                slot = 0
            elif hand_name.lower() == "right":
                slot = 1

        points = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
            dtype=np.float32
        )

        wrist = points[0].copy()
        points = points - wrist

        scale = np.max(np.linalg.norm(points, axis=1))

        if scale > 1e-6:
            points = points / scale

        features[slot] = points

    return features.reshape(FEATURE_COUNT)


def extract_hand_mask(results):
    hand_mask = np.zeros(2, dtype=np.float32)
    if not results.hand_landmarks:
        return hand_mask

    for index in range(min(len(results.hand_landmarks), 2)):
        slot = index
        if (
            results.handedness
            and index < len(results.handedness)
            and results.handedness[index]
        ):
            hand_name = results.handedness[index][0].category_name.lower()
            if hand_name == "left":
                slot = 0
            elif hand_name == "right":
                slot = 1
        hand_mask[slot] = 1.0
    return hand_mask


# --------------------------------------------------
# CAMERA
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    landmarker.close()
    raise RuntimeError("Camera could not be opened!")

sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
prediction_history = deque(maxlen=3)
static_history = deque(maxlen=8)

frame_timestamp = 0
frame_count = 0
last_static_inference_frame = -INFERENCE_INTERVAL
last_dynamic_inference_frame = -INFERENCE_INTERVAL
current_word = "Waiting..."
current_letter = "Waiting..."
confidence = 0.0
static_confidence = 0.0
static_text = ""
pending_static_letter = ""
gesture_had_dynamic_sign = False
gesture_finalized = False
no_hand_since = None
last_static_letter_time = None
voice_lock_active = False
voice_release_since = None
sentence_builder = SentenceBuilder(stable_frames=2, cooldown=0.3)
speech_output = SpeechOutput()

print("\nLive Word Recognition Started!")
print("Show one sign clearly for recognition.")
print("Press BACKSPACE to undo, C to clear, or Q to quit.")

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Unable to read camera frame.")
            break

        frame = cv2.flip(frame, 1)
        frame_timestamp += 33
        frame_count += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        results = landmarker.detect_for_video(
            mp_image,
            frame_timestamp
        )

        current_features = extract_features(results)

        if voice_lock_active:
            if current_features is None and not speech_output.is_busy:
                if voice_release_since is None:
                    voice_release_since = time.monotonic()
                elif time.monotonic() - voice_release_since >= 0.4:
                    voice_lock_active = False
                    voice_release_since = None
                    sequence_buffer.clear()
                    prediction_history.clear()
                    static_history.clear()
                    current_word = "Waiting..."
                    current_letter = "Waiting..."
                    confidence = 0.0
                    static_confidence = 0.0
                    pending_static_letter = ""
                    gesture_had_dynamic_sign = False
                    gesture_finalized = True
                    no_hand_since = None
            else:
                voice_release_since = None

        elif current_features is not None:

            sentence_builder.hand_detected()
            no_hand_since = None
            if gesture_finalized:
                if (
                    last_static_letter_time is not None
                    and time.monotonic() - last_static_letter_time >= 1.2
                    and static_text
                    and not static_text.endswith(" ")
                ):
                    static_text += " "
                pending_static_letter = ""
                gesture_had_dynamic_sign = False
                gesture_finalized = False
                static_history.clear()
            sequence_buffer.append(current_features.copy())

            if frame_count - last_static_inference_frame >= INFERENCE_INTERVAL:
                static_input = torch.tensor(
                    current_features.reshape(1, 42, 3),
                    dtype=torch.float32,
                    device=device,
                )
                static_mask = torch.tensor(
                    extract_hand_mask(results).reshape(1, 2),
                    dtype=torch.float32,
                    device=device,
                )
                with torch.no_grad():
                    static_output = static_model(static_input, adjacency, static_mask)
                    static_probabilities = torch.softmax(static_output, dim=1)
                    static_confidence_value, static_index = torch.max(
                        static_probabilities, dim=1
                    )

                static_prediction = static_labels[static_index.item()]
                static_confidence = static_confidence_value.item() * 100
                static_history.append(static_prediction)
                last_static_inference_frame = frame_count
                if len(static_history) >= 3:
                    stable_letter, stable_count = Counter(static_history).most_common(1)[0]
                    if stable_count >= 3 and static_confidence >= 50:
                        pending_static_letter = stable_letter.upper()
                        current_letter = pending_static_letter
                        confidence = static_confidence

        else:
            sentence_builder.no_hand_detected()
            static_history.clear()
            if no_hand_since is None:
                no_hand_since = time.monotonic()
            elif time.monotonic() - no_hand_since >= HAND_LOSS_RESET_SECONDS and not gesture_finalized:
                if pending_static_letter and not gesture_had_dynamic_sign:
                    static_text += pending_static_letter
                    speech_output.speak(pending_static_letter)
                    last_static_letter_time = time.monotonic()
                    current_word = pending_static_letter
                    confidence = static_confidence
                    voice_lock_active = True
                    voice_release_since = None
                elif not gesture_had_dynamic_sign:
                    current_word = "Waiting..."
                    current_letter = "Waiting..."
                    confidence = 0.0
                    static_confidence = 0.0
                    sequence_buffer.clear()
                    prediction_history.clear()
                pending_static_letter = ""
                gesture_finalized = True

        # --------------------------------------------------
        # WORD PREDICTION
        # --------------------------------------------------

        if (
            current_features is not None
            and len(sequence_buffer) == SEQUENCE_LENGTH
            and not gesture_had_dynamic_sign
            and frame_count - last_dynamic_inference_frame >= INFERENCE_INTERVAL
        ):
            last_dynamic_inference_frame = frame_count

            sequence = np.array(
                sequence_buffer,
                dtype=np.float32
            )

            motion_score = np.abs(np.diff(sequence, axis=0)).mean()

            if motion_score >= DYNAMIC_MOTION_THRESHOLD:
                sample_positions = np.linspace(
                    0, SEQUENCE_LENGTH - 1, MODEL_SEQUENCE_LENGTH
                )
                frame_positions = np.arange(SEQUENCE_LENGTH)
                model_sequence = np.stack(
                    [
                        np.interp(
                            sample_positions,
                            frame_positions,
                            sequence[:, feature_index],
                        )
                        for feature_index in range(FEATURE_COUNT)
                    ],
                    axis=1,
                ).astype(np.float32)
                input_tensor = torch.tensor(
                    model_sequence,
                    dtype=torch.float32
                ).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    confidence_value, predicted_id = torch.max(
                        probabilities, dim=1
                    )

                predicted_word = classes[predicted_id.item()]
                confidence = confidence_value.item() * 100
                prediction_history.append(predicted_word)

                if len(prediction_history) == prediction_history.maxlen:
                    most_common = Counter(prediction_history).most_common(1)[0]
                    if most_common[1] >= 2 and confidence >= 50:
                        current_word = most_common[0]
                        recognized_word = sentence_builder.update(most_common[0])
                        if recognized_word is not None:
                            sentence_builder.add_space()
                            speech_output.speak(sentence_builder.sentence_parts[-1])
                            current_word = recognized_word
                            pending_static_letter = ""
                            gesture_had_dynamic_sign = True
                            voice_lock_active = True
                            voice_release_since = None
            else:
                prediction_history.clear()
                current_word = "Waiting..."

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                points = [
                    (
                        int(landmark.x * frame.shape[1]),
                        int(landmark.y * frame.shape[0]),
                    )
                    for landmark in hand_landmarks
                ]
                for start, end in HAND_CONNECTIONS:
                    cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
                for point in points:
                    cv2.circle(frame, point, 3, (0, 255, 0), -1)

        # --------------------------------------------------
        # DISPLAY
        # --------------------------------------------------

        cv2.putText(
            frame,
            "ISL LIVE WORD RECOGNITION",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Word: {current_word}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Letter: {current_letter}",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.1f}%",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Frames: {len(sequence_buffer)}/{SEQUENCE_LENGTH}",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        sentence = " ".join(
            part for part in (sentence_builder.get_text(), static_text) if part
        )
        sentence_lines = []
        max_line_width = frame.shape[1] - 40
        for sentence_part in re.split(r"(?<=[.!?])\s+", sentence):
            remaining = sentence_part.strip()
            while remaining:
                line = remaining
                while line and cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )[0][0] > max_line_width:
                    split_at = line.rfind(" ")
                    if split_at <= 0:
                        break
                    line = line[:split_at]
                sentence_lines.append(line)
                remaining = remaining[len(line):].strip()

        cv2.putText(
            frame,
            "Sentence:",
            (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        visible_lines = sentence_lines[-5:]
        for line_index, line in enumerate(visible_lines):
            cv2.putText(
                frame,
                line,
                (20, 240 + line_index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        cv2.putText(
            frame,
            "BACKSPACE: undo | C: clear | Q: quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.imshow("ISL Live Word Prediction", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key in (8, 127) and not voice_lock_active:
            if static_text:
                static_text = static_text[:-1]
            else:
                sentence_builder.undo()
            current_word = sentence_builder.current_word or "Waiting..."
            if not sentence_builder.get_text() and not static_text:
                current_word = "Waiting..."

        elif key == ord("c") and not voice_lock_active:
            sentence_builder.clear()
            static_text = ""
            sequence_buffer.clear()
            prediction_history.clear()
            static_history.clear()
            current_word = "Waiting..."
            confidence = 0.0
            static_confidence = 0.0
            pending_static_letter = ""
            gesture_had_dynamic_sign = False
            gesture_finalized = False
            no_hand_since = None
            last_static_letter_time = None

finally:

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    speech_output.close()

    print("\nLive Word Recognition Closed.")
