from model_loader import model, label_encoder, adjacency

print("\nMODEL TEST")

print("Model Loaded:", model is not None)
print("Number of Classes:", len(label_encoder.classes_))
print("Classes:", label_encoder.classes_)
print("Adjacency Shape:", adjacency.shape)
print("Model Parameters:", sum(p.numel() for p in model.parameters()))

print("\nModel Test Completed Successfully!")