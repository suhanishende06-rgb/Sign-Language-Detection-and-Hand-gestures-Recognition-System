
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "dynamic_signs"
)

SEQUENCE_LENGTH = 30
FEATURE_COUNT = 126

print("\n========== DYNAMIC DATASET VERIFICATION ==========\n")

if not os.path.exists(DATASET_DIR):
    print("Dynamic dataset folder not found!")
    raise SystemExit

total_sequences = 0

for label in sorted(os.listdir(DATASET_DIR)):

    label_path = os.path.join(DATASET_DIR, label)

    if not os.path.isdir(label_path):
        continue

    files = [
        f for f in os.listdir(label_path)
        if f.endswith(".npy")
    ]

    print(f"\nSign Label: {label}")
    print(f"Total Sequences: {len(files)}")

    for filename in sorted(files):

        file_path = os.path.join(label_path, filename)

        try:
            sequence = np.load(file_path)

            if sequence.shape != (SEQUENCE_LENGTH, FEATURE_COUNT):
                print(
                    f"INVALID: {filename} "
                    f"Shape: {sequence.shape}"
                )
                continue

            if not np.isfinite(sequence).all():
                print(f"INVALID VALUES: {filename}")
                continue

            total_sequences += 1

        except Exception as error:
            print(f"Error in {filename}: {error}")

print("\n==============================================")
print("Total Valid Sequences:", total_sequences)
print("Expected Shape: (30, 126)")
print("==============================================")

if total_sequences >= 20:
    print("SUCCESS: Sequence dataset verified!")
else:
    print("More sequences are required.")
