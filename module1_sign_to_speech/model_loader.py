
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency):
        x = torch.matmul(adjacency, x)
        return self.linear(x)


class ISL_GCN(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.gcn1 = GCNLayer(3, 32)
        self.gcn2 = GCNLayer(32, 64)
        self.gcn3 = GCNLayer(64, 64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, adjacency, hand_mask):
        x = F.relu(self.gcn1(x, adjacency))
        x = F.relu(self.gcn2(x, adjacency))
        x = F.relu(self.gcn3(x, adjacency))

        node_mask = hand_mask.repeat_interleave(21, dim=1)
        node_mask = node_mask.unsqueeze(-1).float()

        pooled = (x * node_mask).sum(dim=1)
        pooled = pooled / node_mask.sum(dim=1).clamp(min=1)

        return self.classifier(pooled)


device = torch.device("cpu")

model = ISL_GCN(num_classes=26).to(device)

model_path = os.path.join(MODEL_DIR, "isl_gcn_model.pth")

state_dict = torch.load(
    model_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

# Recreate Label Encoder
label_encoder = LabelEncoder()
label_encoder.fit(list("abcdefghijklmnopqrstuvwxyz"))

# Load Adjacency Matrix
adjacency = np.load(
    os.path.join(MODEL_DIR, "adjacency_matrix.npy")
)

adjacency = torch.tensor(
    adjacency, dtype=torch.float32
).to(device)

print("GCN Model Loaded Successfully!")
print("Classes:", list(label_encoder.classes_))
print("Adjacency Shape:", adjacency.shape)
