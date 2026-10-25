import sys
sys.path.append("backend")
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from app.database import get_db
from app.models.pipeline import get_subject_level_partitions, prepare_partition_dataset
from app.models.cnn_transformer import EEGCNNTransformer

with get_db() as conn:
    c = conn.cursor()
    c.execute("SELECT subject_id, diagnostic_label, recording_path FROM subjects WHERE dataset_id='aseeg' AND is_excluded=0 AND file_exists=1")
    rows = [dict(r) for r in c.fetchall()]

train_subs, val_subs, test_subs = get_subject_level_partitions(rows, test_ratio=0.25, val_ratio=0.15, random_seed=42)
print(f"Train: {len(train_subs)}, Val: {len(val_subs)}, Test: {len(test_subs)}")

t0 = time.time()
X_train, y_train, _, norm_stats = prepare_partition_dataset(train_subs, "SZ_VS_HC")
print(f"Prepared train in {time.time()-t0:.2f}s, X_train shape: {X_train.shape}")

t0 = time.time()
X_val, y_val, _, _ = prepare_partition_dataset(val_subs, "SZ_VS_HC", norm_stats=norm_stats)
print(f"Prepared val in {time.time()-t0:.2f}s, X_val shape: {X_val.shape}")

model = EEGCNNTransformer(num_channels=16, time_points=500, num_classes=2, d_model=64)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=32, shuffle=False)

for epoch in range(1, 4):
    t0 = time.time()
    model.train()
    total_loss = 0
    for bx, by in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(bx)
    train_loss = total_loss / len(X_train)

    model.eval()
    correct = 0
    with torch.no_grad():
        for bx, by in val_loader:
            preds = model(bx).argmax(dim=-1)
            correct += int((preds == by).sum().item())
    val_acc = correct / len(X_val)
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Acc = {val_acc:.4f}, Time = {time.time()-t0:.2f}s")
