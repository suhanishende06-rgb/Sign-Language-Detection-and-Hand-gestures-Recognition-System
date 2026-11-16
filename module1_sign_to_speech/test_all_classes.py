
import os
import cv2
import numpy as np
import torch
import mediapipe as mp

from model_loader import model, label_encoder, adjacency, device


# 1. Paths

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_DIR = os.path.join(
    BASE_DIR, "dataset", "static_signs"
)

TASK_PATH = os.path.join(
    BASE_DIR, "hand_landmarker.task"
)

SAMPLES_PER_CLASS = 20


# 2. Validate paths

if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )

if not os.path.isfile(TASK_PATH):
    raise FileNotFoundError(
        f"Hand Landmarker model not found: {TASK_PATH}"
    )


# 3. Initialize MediaPipe

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=TASK_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)

detector = mp.tasks.vision.HandLandmarker.create_from_options(
    options
)


# 4. Prediction function

def predict_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None, None, "Image loading failed"

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    results = detector.detect(mp_image)

    if not results.hand_landmarks:
        return None, None, "No hand detected"

    features = np.zeros(
        (2, 21, 3),
        dtype=np.float32
    )

    hand_mask = np.zeros(
        2,
        dtype=np.float32
    )

    for i, hand in enumerate(results.hand_landmarks[:2]):

        for j, landmark in enumerate(hand):

            features[i, j] = [
                landmark.x,
                landmark.y,
                landmark.z
            ]

        hand_mask[i] = 1.0

    # Normalize landmarks

    for hand in range(2):

        if hand_mask[hand] == 1:

            wrist = features[hand, 0].copy()

            features[hand] -= wrist

            distances = np.linalg.norm(
                features[hand],
                axis=1
            )

            scale = np.max(distances)

            if scale > 0:
                features[hand] /= scale

    input_tensor = torch.tensor(
        features.reshape(1, 42, 3),
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

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    predicted_label = label_encoder.inverse_transform(
        [predicted.item()]
    )[0].upper()

    confidence_value = confidence.item() * 100

    return predicted_label, confidence_value, "OK"


# 5. Test all classes

results_summary = []

total_correct = 0
total_wrong = 0
total_valid = 0
total_no_hand = 0
total_failed = 0

print("\n========== A TO Z CLASS TESTING ==========\n")

try:

    for actual_label in "abcdefghijklmnopqrstuvwxyz":

        class_folder = os.path.join(
            DATASET_DIR,
            actual_label
        )

        if not os.path.isdir(class_folder):

            print(f"{actual_label.upper()}: Folder not found")
            continue

        images = sorted([
            f for f in os.listdir(class_folder)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])

        correct = 0
        wrong = 0
        no_hand = 0
        failed = 0
        tested = 0

        print(f"\n---------- {actual_label.upper()} CLASS ----------")

        for filename in images:

            if tested >= SAMPLES_PER_CLASS:
                break

            image_path = os.path.join(
                class_folder,
                filename
            )

            predicted, confidence, status = predict_image(
                image_path
            )

            if status == "No hand detected":
                no_hand += 1
                continue

            if status != "OK":
                failed += 1
                continue

            tested += 1

            if predicted == actual_label.upper():

                correct += 1
                result = "CORRECT"

            else:

                wrong += 1
                result = "WRONG"

            print(
                f"{filename}: "
                f"Predicted={predicted}, "
                f"Confidence={confidence:.2f}%, "
                f"{result}"
            )

        accuracy = (
            (correct / tested) * 100
            if tested > 0 else 0
        )

        total_correct += correct
        total_wrong += wrong
        total_valid += tested
        total_no_hand += no_hand
        total_failed += failed

        results_summary.append({
            "Class": actual_label.upper(),
            "Tested": tested,
            "Correct": correct,
            "Wrong": wrong,
            "No Hand": no_hand,
            "Accuracy": accuracy
        })

        print(
            f"{actual_label.upper()} Accuracy: "
            f"{accuracy:.2f}% "
            f"({correct}/{tested})"
        )


finally:

    detector.close()


# 6. Final report

print("\n========== FINAL A TO Z REPORT ==========")

print(
    f"{'CLASS':<10}"
    f"{'TESTED':<10}"
    f"{'CORRECT':<10}"
    f"{'WRONG':<10}"
    f"{'ACCURACY':<12}"
)

for item in results_summary:

    print(
        f"{item['Class']:<10}"
        f"{item['Tested']:<10}"
        f"{item['Correct']:<10}"
        f"{item['Wrong']:<10}"
        f"{item['Accuracy']:.2f}%"
    )

overall_accuracy = (
    (total_correct / total_valid) * 100
    if total_valid > 0 else 0
)

print("\nTotal Valid Images:", total_valid)
print("Total Correct:", total_correct)
print("Total Wrong:", total_wrong)
print("Total No Hand Detected:", total_no_hand)
print("Total Image Loading Failed:", total_failed)

print(
    "Overall Accuracy:",
    round(overall_accuracy, 2),
    "%"
)

print("=========================================")
