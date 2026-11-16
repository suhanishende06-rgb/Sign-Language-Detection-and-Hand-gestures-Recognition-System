import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATASET_PATH = "dataset/static_signs"

os.makedirs(DATASET_PATH, exist_ok=True)

label = input("Enter Sign Label (A-Z or 1-9): ").upper()

file_path = os.path.join(DATASET_PATH, f"{label}.csv")

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

camera = cv2.VideoCapture(0)

with open(file_path, "w", newline="") as file:
    writer = csv.writer(file)

    while camera.isOpened():
        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                row = []

                for landmark in hand_landmarks.landmark:
                    row.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z
                    ])

                writer.writerow(row)

        cv2.putText(
            frame,
            f"Collecting Sign: {label} | Press Q to Stop",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.imshow("ISL Dataset Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
hands.close()
cv2.destroyAllWindows()

print(f"Dataset saved successfully: {file_path}")