
import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# Settings
DATASET_PATH = "dataset/static_signs"
FRAMES_PER_SAMPLE = 30

os.makedirs(DATASET_PATH, exist_ok=True)

# Get person name and sign label
person_id = input("Enter Person ID (e.g. Suhani / Friend1): ").strip()

if not person_id:
    print("Person ID cannot be empty!")
    raise SystemExit

label = input("Enter Sign Label (A-Z or 1-9): ").strip().upper()

if len(label) != 1 or not (("A" <= label <= "Z") or ("1" <= label <= "9")):
    print("Invalid label! Enter A-Z or 1-9.")
    raise SystemExit

file_path = os.path.join(DATASET_PATH, f"{label}_sequences.csv")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not detected!")
    hands.close()
    raise SystemExit

# CSV header
header = ["label", "person_id", "sample_id", "frame_id"]

for hand in ["left", "right"]:
    for point in range(21):
        for coordinate in ["x", "y", "z"]:
            header.append(f"{hand}_{point}_{coordinate}")

# Preserve and migrate old CSV if required
if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    with open(file_path, "r", newline="") as file:
        reader = csv.reader(file)
        old_header = next(reader, [])
        old_rows = list(reader)

    old_header_without_person = [
        "label", "sample_id", "frame_id"
    ] + header[4:]

    if old_header == old_header_without_person:
        # Add person_id to previously collected samples
        migrated_rows = []

        for row in old_rows:
            if len(row) == len(old_header):
                migrated_rows.append([
                    row[0], "Suhani", *row[1:]
                ])

        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(migrated_rows)

        print("Old dataset updated. Previous samples preserved!")

    elif old_header != header:
        print("CSV format mismatch! Please check the existing dataset.")
        camera.release()
        hands.close()
        cv2.destroyAllWindows()
        raise SystemExit

# Find next sample ID
sample_id = 1

if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    with open(file_path, "r", newline="") as file:
        reader = csv.DictReader(file)
        existing_ids = [
            int(row["sample_id"])
            for row in reader
            if row.get("sample_id", "").isdigit()
        ]

        if existing_ids:
            sample_id = max(existing_ids) + 1

# Append mode: old samples will not be deleted
with open(file_path, "a", newline="") as file:
    writer = csv.writer(file)

    if os.path.getsize(file_path) == 0:
        writer.writerow(header)

    frame_count = 0
    collecting = False
    sample_data = []

    print("Press S to collect 30 frames.")
    print("Press Q to exit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Camera frame not available!")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        landmarks_data = {
            "Left": np.zeros(63),
            "Right": np.zeros(63)
        }

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                side = handedness.classification[0].label

                coords = []

                for point in hand_landmarks.landmark:
                    coords.extend([point.x, point.y, point.z])

                landmarks_data[side] = np.array(coords)

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        if collecting:
            combined = np.concatenate([
                landmarks_data["Left"],
                landmarks_data["Right"]
            ])

            sample_data.append(combined)
            frame_count += 1

            if frame_count == FRAMES_PER_SAMPLE:
                for frame_id, values in enumerate(sample_data, start=1):
                    writer.writerow([
                        label,
                        person_id,
                        sample_id,
                        frame_id,
                        *values
                    ])

                file.flush()

                print(f"Sample {sample_id} saved for {person_id}!")

                sample_id += 1
                frame_count = 0
                sample_data = []
                collecting = False

        cv2.putText(
            frame,
            f"Sign: {label} | Person: {person_id} | Sample: {sample_id}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        status = "COLLECTING" if collecting else "Press S to Start"

        cv2.putText(
            frame,
            status,
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.imshow("ISL Dataset Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") and not collecting:
            collecting = True
            frame_count = 0
            sample_data = []
            print("Collecting 30 frames...")

        elif key == ord("q"):
            break

camera.release()
hands.close()
cv2.destroyAllWindows()

print("Dataset collection completed!")
print("Saved file:", file_path)
