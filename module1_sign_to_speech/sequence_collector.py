
import os
import cv2
import time
import numpy as np
import mediapipe as mp

from datetime import datetime

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASK_MODEL = os.path.join(BASE_DIR, "hand_landmarker.task")

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "dynamic_signs"
)

SEQUENCE_LENGTH = 30
FEATURE_COUNT = 126

os.makedirs(DATASET_DIR, exist_ok=True)

# --------------------------------------------------
# SIGN LABEL
# --------------------------------------------------

label = input(
    "Enter dynamic sign label (example: HELLO): "
).strip().upper()

if not label:
    print("Invalid label!")
    raise SystemExit

label = "".join(
    ch if ch.isalnum() or ch in "_-" else "_"
    for ch in label
)

LABEL_DIR = os.path.join(DATASET_DIR, label)
os.makedirs(LABEL_DIR, exist_ok=True)

# --------------------------------------------------
# MEDIAPIPE HAND LANDMARKER
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=TASK_MODEL
    ),
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

def extract_features(results):

    features = np.zeros(
        (2, 21, 3),
        dtype=np.float32
    )

    hand_mask = np.zeros(2, dtype=np.float32)

    if not results.hand_landmarks:
        return None

    for index, hand_landmarks in enumerate(
        results.hand_landmarks[:2]
    ):

        # Assign landmarks to left/right hand slots
        slot = index

        if (
            results.handedness
            and index < len(results.handedness)
            and results.handedness[index]
        ):
            hand_name = (
                results.handedness[index][0].category_name
            )

            if hand_name.lower() == "left":
                slot = 0
            elif hand_name.lower() == "right":
                slot = 1

        points = np.array(
            [
                [lm.x, lm.y, lm.z]
                for lm in hand_landmarks
            ],
            dtype=np.float32
        )

        # Wrist normalization
        wrist = points[0].copy()
        points = points - wrist

        scale = np.max(
            np.linalg.norm(points, axis=1)
        )

        if scale > 1e-6:
            points = points / scale

        features[slot] = points
        hand_mask[slot] = 1.0

    if hand_mask.sum() == 0:
        return None

    return features.reshape(FEATURE_COUNT)


# --------------------------------------------------
# CAMERA
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened!")
    landmarker.close()
    raise SystemExit

WINDOW_NAME = "ISL Dynamic Sequence Collector"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

frame_timestamp = 0

recording = False
sequence_frames = []

saved_count = len([
    f for f in os.listdir(LABEL_DIR)
    if f.endswith(".npy")
])

print("\nDynamic Sequence Collector Started!")
print("Label:", label)
print("Existing sequences:", saved_count)

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

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        results = landmarker.detect_for_video(
            mp_image,
            frame_timestamp
        )

        current_features = extract_features(results)

        # ------------------------------------------
        # DRAW LANDMARKS
        # ------------------------------------------

        if results.hand_landmarks:

            for hand_landmarks in results.hand_landmarks:

                points = []

                for landmark in hand_landmarks:

                    x = int(
                        landmark.x * frame.shape[1]
                    )

                    y = int(
                        landmark.y * frame.shape[0]
                    )

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
        # RECORD 30 VALID HAND FRAMES
        # ------------------------------------------

        if recording:

            if current_features is not None:

                sequence_frames.append(
                    current_features.copy()
                )

                if len(sequence_frames) >= SEQUENCE_LENGTH:

                    sequence = np.array(
                        sequence_frames[:SEQUENCE_LENGTH],
                        dtype=np.float32
                    )

                    filename = (
                        f"{label}_{int(time.time() * 1000)}.npy"
                    )

                    save_path = os.path.join(
                        LABEL_DIR,
                        filename
                    )

                    np.save(save_path, sequence)

                    saved_count += 1

                    print(
                        f"Sequence {saved_count} saved: "
                        f"{save_path}"
                    )

                    sequence_frames = []
                    recording = False

            else:

                cv2.putText(
                    frame,
                    "Show your hand clearly!",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        # ------------------------------------------
        # DISPLAY INFORMATION
        # ------------------------------------------

        cv2.putText(
            frame,
            "ISL Dynamic Sequence Collector",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Sign Label: {label}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Saved Sequences: {saved_count}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        progress = len(sequence_frames)

        status = (
            f"Recording: {progress}/{SEQUENCE_LENGTH}"
            if recording
            else "Press R to Record"
        )

        cv2.putText(
            frame,
            status,
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        controls = "R - Record    Q - Quit"

        cv2.putText(
            frame,
            controls,
            (15, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.imshow(WINDOW_NAME, frame)

        # ------------------------------------------
        # KEYBOARD CONTROLS
        # ------------------------------------------

        key = cv2.waitKeyEx(10)

        if key != -1:

            key = key & 0xFF

            if key == ord("q") or key == ord("Q"):

                print("Closing collector...")
                break

            elif key == ord("r") or key == ord("R"):

                if not recording:

                    sequence_frames = []
                    recording = True

                    print(
                        "Recording started. Perform the sign!"
                    )

                else:

                    print("Already recording!")

finally:

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

    print("\nSequence Collector Closed!")
    print("Label:", label)
    print("Total Saved Sequences:", saved_count)
