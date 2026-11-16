
import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(BASE_DIR, "dataset", "holistic_signs")
os.makedirs(SAVE_DIR, exist_ok=True)

# MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Settings
SEQUENCE_LENGTH = 30
FEATURE_COUNT = 188

# Selected upper-body landmarks:
# Shoulders, elbows, wrists and hips
POSE_IDS = [11, 12, 13, 14, 15, 16, 23, 24]

# Selected facial landmarks:
# Nose, lips, mouth corners, eyebrows and eyes
FACE_IDS = [1, 13, 14, 61, 291, 70, 300, 159, 386, 10]


def extract_features(results):
    features = []

    # 1. LEFT HAND: 21 x 3 = 63
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 63)

    # 2. RIGHT HAND: 21 x 3 = 63
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 63)

    # 3. BODY: 8 x 4 = 32
    if results.pose_landmarks:
        for idx in POSE_IDS:
            lm = results.pose_landmarks.landmark[idx]
            features.extend([
                lm.x, lm.y, lm.z, lm.visibility
            ])
    else:
        features.extend([0.0] * 32)

    # 4. FACE: 10 x 3 = 30
    if results.face_landmarks:
        for idx in FACE_IDS:
            lm = results.face_landmarks.landmark[idx]
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 30)

    return np.array(features, dtype=np.float32)


def main():
    label = input("Enter ISL Sign Label (e.g. HELLO): ")
    label = label.strip().upper()

    if not label:
        print("Invalid label!")
        return

    label_dir = os.path.join(SAVE_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    existing = [
        f for f in os.listdir(label_dir)
        if f.endswith(".npy")
    ]

    sequence_number = len(existing) + 1

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Camera not detected!")
        return

    recording = False
    sequence = []
    saved_count = 0

    print("\nCONTROLS")
    print("R = Record one 30-frame sequence")
    print("Q = Quit")
    print("Show the complete upper body, face and hands.")

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        while camera.isOpened():
            success, frame = camera.read()

            if not success:
                print("Camera frame error!")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            results = holistic.process(rgb)

            rgb.flags.writeable = True

            # Draw body
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS
                )

            # Draw hands
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )

            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )

            # Draw face
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None
                )

            # Status
            face_ok = results.face_landmarks is not None
            body_ok = results.pose_landmarks is not None
            hand_ok = (
                results.left_hand_landmarks is not None
                or results.right_hand_landmarks is not None
            )

            status = (
                f"Face:{face_ok} Body:{body_ok} Hands:{hand_ok}"
            )

            cv2.putText(
                frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0), 2
            )

            cv2.putText(
                frame,
                f"Label: {label} | Saved: {saved_count}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 2
            )

            if recording:
                cv2.putText(
                    frame,
                    f"RECORDING: {len(sequence)}/{SEQUENCE_LENGTH}",
                    (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2
                )

                # Record only when face, body and at least one hand exist
                if face_ok and body_ok and hand_ok:
                    feature_vector = extract_features(results)

                    if (
                        len(feature_vector) == FEATURE_COUNT
                        and np.isfinite(feature_vector).all()
                    ):
                        sequence.append(feature_vector)

                if len(sequence) >= SEQUENCE_LENGTH:
                    data = np.array(sequence, dtype=np.float32)

                    file_path = os.path.join(
                        label_dir,
                        f"{label}_{sequence_number:03d}.npy"
                    )

                    np.save(file_path, data)

                    print("Saved:", file_path)

                    sequence_number += 1
                    saved_count += 1
                    sequence = []
                    recording = False

            cv2.imshow("ISL Holistic Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("r") and not recording:
                sequence = []
                recording = True
                print("Recording started. Perform the sign.")

    camera.release()
    cv2.destroyAllWindows()

    print("\nCollection completed.")
    print("Sequences saved:", saved_count)


if __name__ == "__main__":
    main()
