import os
import logging
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from dotenv import load_dotenv
load_dotenv()


DATA_DIR = os.getenv("DATA_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
MODEL_PATH = os.getenv("MODEL_PATH")
LOGGING_PATH = os.getenv("LOGGING_PATH")
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(
    filename=LOGGING_PATH,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()
logger.info("Using device: %s", DEVICE)


class BoneScanDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        csv_path = os.path.join(folder_path, "_classes_copy.csv")
        self.data = pd.read_csv(csv_path)
        self.folder_path = folder_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_name = row["filename"]
        img_path = os.path.join(self.folder_path, img_name)
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(int(np.argmax([row["cancer"], row["normal"]])))
        if self.transform:
            image = self.transform(image)
        return image, label

# Transforms + DataLoader
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

valid_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_dataset = BoneScanDataset(os.path.join(DATA_DIR, "train"), train_tfms)
valid_dataset = BoneScanDataset(os.path.join(DATA_DIR, "valid"), valid_tfms)
test_dataset  = BoneScanDataset(os.path.join(DATA_DIR, "test"),  valid_tfms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

logger.info(f"Train: {len(train_dataset)} | Valid: {len(valid_dataset)} | Test: {len(test_dataset)}")

# Model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: cancer / normal
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training Loop
train_losses, valid_losses, train_accs, valid_accs = [], [], [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_losses.append(total_loss / len(train_loader))
    train_accs.append(correct / total)

    # Validation
    model.eval()
    v_loss, v_correct, v_total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            v_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            v_correct += (preds == labels).sum().item()
            v_total += labels.size(0)
    valid_losses.append(v_loss / len(valid_loader))
    valid_accs.append(v_correct / v_total)

    logger.info(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_losses[-1]:.4f} | Val Loss: {valid_losses[-1]:.4f} | Val Acc: {valid_accs[-1]*100:.2f}%")

# Plot training history
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train')
plt.plot(valid_losses, label='Valid')
plt.title('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs, label='Train')
plt.plot(valid_accs, label='Valid')
plt.title('Accuracy')
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/training_history.png")

# Save model
torch.save(model.state_dict(), MODEL_PATH)
print("Model saved as model.pth")

# Testing + Evaluation
model.eval()
all_labels, all_preds, all_probs = [], [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)[:,1]  # probability of class 1 (normal)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())


# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["cancer", "normal"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")

# ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/roc_curve.png")
