import torch
import torch.nn as nn
import numpy as np
from glob import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from model import DroneClassifier

# === Параметры ===
DATA_DIR = "fpv_dataset"
INPUT_LEN = 256
CLASSES = ["DJI", "FPV", "None"]
BATCH_SIZE = 64
EPOCHS = 20

# === Загрузка данных ===
X, y = [], []
for path in glob(os.path.join(DATA_DIR, "*.npz")):
    data = np.load(path)
    ampl = data["ampl"]
    if len(ampl) != INPUT_LEN:
        continue
    spectrum = np.log10(np.clip(ampl, 1e-10, None))  # лог-преобразование
    X.append(spectrum)
    y.append(int(data["label"]))

X = np.array(X)  # оптимизация: быстрее, чем список
y = np.array(y)

X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # [N, 1, 256]
y = torch.tensor(y, dtype=torch.long)

# === Train/test split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === DataLoaders ===
train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# === Веса классов (если дисбаланс) ===
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y.numpy()), y=y.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# === Модель, оптимизатор, лосс ===
model = DroneClassifier(num_classes=len(CLASSES))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# === Обучение ===
print("🔧 Обучение модели начато...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

    # === Оценка ===
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            out = model(xb)
            preds = torch.argmax(out, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    acc = correct / total
    print(f"Epoch {epoch+1:02d} | Loss: {total_loss / len(train_ds):.4f} | Val Acc: {acc:.3f}")

# === Сохранение ===
torch.save(model.state_dict(), "fpv_model.pt")
print("✅ Модель сохранена как fpv_model.pt")
