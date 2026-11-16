import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIRS = [
    BASE_DIR / "dataset" / "dynamic_signs",
    BASE_DIR / "dataset" / "include_dynamic",
]
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "isl_word_bilstm_include.pth"
LABEL_PATH = MODEL_DIR / "word_labels_include.json"
SEQUENCE_SHAPE = (30, 126)


class WordBiLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=126,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(self.dropout(output[:, -1, :]))


def load_sequences():
    sequences, labels = [], []
    for data_dir in DATA_DIRS:
        if not data_dir.is_dir():
            continue
        for label_dir in sorted(data_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name.upper()
            for file_path in sorted(label_dir.glob("*.npy")):
                sequence = np.load(file_path)
                if sequence.shape != SEQUENCE_SHAPE or not np.isfinite(sequence).all():
                    print(f"Skipped invalid sequence: {file_path}")
                    continue
                sequences.append(sequence.astype(np.float32))
                labels.append(label)
    if not sequences:
        raise RuntimeError(
            "No valid sequences found. Collect dynamic signs or run "
            "include_dynamic_dataset.py extract first."
        )
    return np.stack(sequences), labels


def split_by_class(labels, seed=42):
    rng = random.Random(seed)
    by_label = {}
    for index, label in enumerate(labels):
        by_label.setdefault(label, []).append(index)
    train_indices, test_indices = [], []
    for indexes in by_label.values():
        rng.shuffle(indexes)
        if len(indexes) < 2:
            train_indices.extend(indexes)
            continue
        test_count = max(1, round(len(indexes) * 0.2))
        test_count = min(test_count, len(indexes) - 1)
        test_indices.extend(indexes[:test_count])
        train_indices.extend(indexes[test_count:])
    if not test_indices:
        raise RuntimeError("At least one word must have two or more sequences for validation.")
    return np.asarray(train_indices), np.asarray(test_indices)


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cpu")

    x, labels = load_sequences()
    classes = sorted(set(labels))
    label_to_id = {label: index for index, label in enumerate(classes)}
    y = np.asarray([label_to_id[label] for label in labels], dtype=np.int64)
    train_indices, test_indices = split_by_class(labels)
    print(
        f"Sequences: {len(x)} | Classes: {len(classes)} | "
        f"Train: {len(train_indices)} | Validation: {len(test_indices)}"
    )

    train_data = TensorDataset(
        torch.from_numpy(x[train_indices]),
        torch.from_numpy(y[train_indices]),
    )
    test_data = TensorDataset(
        torch.from_numpy(x[test_indices]),
        torch.from_numpy(y[test_indices]),
    )
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16)

    counts = np.bincount(y[train_indices], minlength=len(classes))
    class_weights = np.sqrt(counts.max() / np.maximum(counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    model = WordBiLSTM(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    best_accuracy = -1.0
    epochs_without_improvement = 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(30):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                predictions = model(inputs.to(device)).argmax(dim=1)
                correct += (predictions == targets.to(device)).sum().item()
                total += targets.size(0)
        accuracy = correct / total
        print(f"Epoch {epoch + 1:02d}/30 | Validation accuracy: {accuracy * 100:.2f}%")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
            LABEL_PATH.write_text(json.dumps(classes, indent=2), encoding="utf-8")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 6:
                break

    print(f"Best validation accuracy: {best_accuracy * 100:.2f}%")
    print(f"New model saved to: {MODEL_PATH}")
    print("The original isl_word_bilstm.pth and word_labels.json are not modified.")


if __name__ == "__main__":
    main()
