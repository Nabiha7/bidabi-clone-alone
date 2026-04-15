import os
import copy
import random
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

# -------------------------
# Paramètres
# -------------------------
DATA_DIR = "data/raw/images"
BATCH_SIZE = 16
EPOCHS = 5
MODEL_PATH = "model_best.pth"
SEED = 42

# -------------------------
# Reproductibilité
# -------------------------
random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# Dataset de base
# -------------------------
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

full_dataset = datasets.ImageFolder(DATA_DIR, transform=base_transform)

train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_set, val_set, test_set = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)

# -------------------------
# Augmentations
# -------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_set.dataset.transform = train_transform
val_set.dataset.transform = eval_transform
test_set.dataset.transform = eval_transform

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# Modèle
# -------------------------
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, len(full_dataset.classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_val_loss = float("inf")
best_model_wts = copy.deepcopy(model.state_dict())

# -------------------------
# Fonctions utilitaires
# -------------------------
def evaluate(loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy

# -------------------------
# Entraînement
# -------------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / max(len(train_loader), 1)
    val_loss, val_acc = evaluate(val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())

# -------------------------
# Sauvegarde du meilleur modèle
# -------------------------
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Meilleur modèle sauvegardé : {MODEL_PATH}")

# -------------------------
# Évaluation finale sur test
# -------------------------
test_loss, test_acc = evaluate(test_loader)
print(f"✅ Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
