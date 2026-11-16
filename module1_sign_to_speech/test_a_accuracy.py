
import os
import cv2
import numpy as np
import torch
import mediapipe as mp

from model_loader import model, label_encoder, adjacency, device


# 1. Dataset and model paths

DATASET_PATH = r"D:\Sign_Language_Recognition_System\dataset\static_signs\a"

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TASK_PATH = os.path.join(
    BASE_DIR,
    "hand_landmarker.task"
)


# 2. Check paths

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("A Dataset folder not found!")

if not os.path.exists(TASK_PATH):
    raise FileNotFoundError("hand_landmarker.task not found!")


# 3. Initialize MediaPipe

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=TASK_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2
)

detector = mp.tasks.vision.HandLandmarker.create_from_options(
    options
)


# 4. Get image files

images = sorted([
    f for f in os.listdir(DATASET_PATH)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

correct = 0
wrong = 0
failed = 0
tested = 0

print("\n========== A CLASS VALID IMAGE TEST ==========")


# 5. Test until 20 valid images

try:

    for filename in images:

        if tested >= 20:
            break

        image_path = os.path.join(
            DATASET_PATH,
            filename
        )

        image = cv2.imread(image_path)

        if image is None:
            failed += 1
            print(filename, "- Image loading failed")
            continue

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        results = detector.detect(mp_image)

        # Check hand detection

        if not results.hand_landmarks:

            failed += 1
            print(filename, "- No hand detected")
            continue

        # Extract landmarks

        features = np.zeros(
            (2, 21, 3),
            dtype=np.float32
        )

        hand_mask = np.zeros(
            2,
            dtype=np.float32
        )

        for i, hand in enumerate(
            results.hand_landmarks[:2]
        ):

            for j, landmark in enumerate(hand):

                features[i, j] = [
                    landmark.x,
                    landmark.y,
                    landmark.z
                ]

            hand_mask[i] = 1.0

        # Normalize landmarks

        for hand in range(2):

            if np.any(features[hand]):

                wrist = features[hand, 0].copy()

                features[hand] -= wrist

                distances = np.linalg.norm(
                    features[hand],
                    axis=1
                )

                scale = np.max(distances)

                if scale > 0:

                    features[hand] /= scale

        # Prepare tensors

        input_tensor = torch.tensor(
            features.reshape(1, 42, 3),
            dtype=torch.float32
        ).to(device)

        mask_tensor = torch.tensor(
            hand_mask.reshape(1, 2),
            dtype=torch.float32
        ).to(device)

        # GCN prediction

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

        tested += 1

        # Compare actual and predicted labels

        if predicted_label == "A":

            correct += 1
            status = "CORRECT"

        else:

            wrong += 1
            status = "WRONG"

        print(
            f"{filename}: "
            f"Predicted={predicted_label}, "
            f"Confidence={confidence_value:.2f}%, "
            f"{status}"
        )


finally:

    detector.close()


# 6. Final results

print("\n========== FINAL RESULTS ==========")

print("Total Images Available:", len(images))
print("Valid Images Tested:", tested)
print("Correct Predictions:", correct)
print("Wrong Predictions:", wrong)
print("Images Without Hand:", failed)

if tested > 0:

    accuracy = (correct / tested) * 100

    print("A Class Accuracy:", round(accuracy, 2), "%")

else:

    print("No valid images detected!")

print("===================================")
