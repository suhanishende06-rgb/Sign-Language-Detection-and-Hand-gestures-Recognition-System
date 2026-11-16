
import cv2
import numpy as np
import torch
import os

from model_loader import model, label_encoder, adjacency, device


# Dataset A folder
DATASET_PATH = r"D:\Sign_Language_Recognition_System\dataset\static_signs\a"

# Select first image
images = [
    file for file in os.listdir(DATASET_PATH)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not images:
    raise FileNotFoundError("No images found in A folder!")

image_path = os.path.join(DATASET_PATH, sorted(images)[0])

print("Testing Image:", image_path)


# Read image
image = cv2.imread(image_path)

if image is None:
    raise ValueError("Image could not be loaded!")


# Convert image to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# MediaPipe Tasks Hand Detection
import mediapipe as mp

base_options = mp.tasks.BaseOptions
hand_landmarker = mp.tasks.vision.HandLandmarker
options_class = mp.tasks.vision.HandLandmarkerOptions
running_mode = mp.tasks.vision.RunningMode

model_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "hand_landmarker.task"
)

options = options_class(
    base_options=base_options(model_asset_path=model_path),
    running_mode=running_mode.IMAGE,
    num_hands=2
)

detector = hand_landmarker.create_from_options(options)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_image
)

results = detector.detect(mp_image)
detector.close()


# Extract landmarks
features = np.zeros((2, 21, 3), dtype=np.float32)
hand_mask = np.zeros(2, dtype=np.float32)

if results.hand_landmarks:

    for i, hand in enumerate(results.hand_landmarks[:2]):

        for j, landmark in enumerate(hand):

            features[i, j] = [
                landmark.x,
                landmark.y,
                landmark.z
            ]

        hand_mask[i] = 1.0

else:
    raise RuntimeError("No hand detected in this image!")


# Normalize landmarks
for hand in range(2):

    if np.any(features[hand]):

        wrist = features[hand, 0].copy()
        features[hand] -= wrist

        distances = np.linalg.norm(features[hand], axis=1)
        scale = np.max(distances)

        if scale > 0:
            features[hand] /= scale


# GCN prediction
input_tensor = torch.tensor(
    features.reshape(1, 42, 3),
    dtype=torch.float32
).to(device)

mask_tensor = torch.tensor(
    hand_mask.reshape(1, 2),
    dtype=torch.float32
).to(device)

with torch.no_grad():

    outputs = model(input_tensor, adjacency, mask_tensor)

    probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probabilities, dim=1)

predicted_letter = label_encoder.inverse_transform(
    [predicted.item()]
)[0]


print("\n========== DATASET IMAGE TEST ==========")
print("Actual Label: A")
print("Predicted Label:", predicted_letter.upper())
print("Confidence:", round(confidence.item() * 100, 2), "%")
print("========================================")


# Show image
cv2.imshow("Dataset A Image Test", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
