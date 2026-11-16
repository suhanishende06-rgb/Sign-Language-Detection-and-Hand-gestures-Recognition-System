
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "dynamic_signs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cpu")

# Load sequences
X, labels = [], []

for word in sorted(os.listdir(DATA_DIR)):
    folder = os.path.join(DATA_DIR, word)

    if not os.path.isdir(folder):
        continue

    for filename in os.listdir(folder):
        if filename.endswith(".npy"):
            sequence = np.load(os.path.join(folder, filename))

            if sequence.shape != (30, 126):
                print("Skipped:", filename, sequence.shape)
                continue

            X.append(sequence.astype(np.float32))
            labels.append(word)

X = np.array(X, dtype=np.float32)
classes = sorted(set(labels))
label_to_id = {name: i for i, name in enumerate(classes)}
y = np.array([label_to_id[label] for label in labels])

print("Dataset:", X.shape)
print("Classes:", classes)

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

train_dataset = TensorDataset(
    torch.tensor(X_train),
    torch.tensor(y_train, dtype=torch.long)
)

test_dataset = TensorDataset(
    torch.tensor(X_test),
    torch.tensor(y_test, dtype=torch.long)
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# BiLSTM model
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

model = WordBiLSTM(len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001, weight_decay=0.0001
)

best_accuracy = 0.0

# Training
for epoch in range(50):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == targets).sum().item()
        total += targets.size(0)

    train_accuracy = 100 * correct / total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs.to(device))
            predictions = outputs.argmax(1)
            val_correct += (predictions == targets.to(device)).sum().item()
            val_total += targets.size(0)

    val_accuracy = 100 * val_correct / val_total

    print(
        f"Epoch {epoch+1:02d}/50 | "
        f"Loss: {total_loss/len(train_loader):.4f} | "
        f"Train: {train_accuracy:.2f}% | "
        f"Test: {val_accuracy:.2f}%"
    )

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(),
                   os.path.join(MODEL_DIR, "isl_word_bilstm.pth"))

        with open(os.path.join(MODEL_DIR, "word_labels.json"), "w") as f:
            json.dump(classes, f)

print("\nTraining Completed!")
print("Best Test Accuracy:", round(best_accuracy, 2), "%")
print("Model saved in:", MODEL_DIR)
