# -----------------------------------
# 1. Load and Visualize Epochs
# -----------------------------------
import os
import mne
import matplotlib.pyplot as plt
import numpy as np

processed_folder = 'processed_data'
epoch_files = [f for f in os.listdir(processed_folder) if f.endswith('-epo.fif')]

all_standard = []
all_oddball = []

for epo_file in epoch_files:
    print(f"Processing {epo_file}")
    epochs = mne.read_epochs(os.path.join(processed_folder, epo_file), preload=True)
    
    standard_keys = [k for k in epochs.event_id if 'standard' in k.lower()]
    oddball_keys = [k for k in epochs.event_id if 'oddball' in k.lower() or 'target' in k.lower()]
    
    if not standard_keys or not oddball_keys:
        print(f"Could not find standard or oddball events in {epo_file}. Skipping.")
        continue
    
    epochs_standard = epochs[standard_keys]
    epochs_oddball = epochs[oddball_keys]
    
    all_standard.append(epochs_standard.get_data())
    all_oddball.append(epochs_oddball.get_data())

if all_standard and all_oddball:
    combined_standard = np.concatenate(all_standard, axis=0)
    combined_oddball = np.concatenate(all_oddball, axis=0)
else:
    raise ValueError("No standard or oddball trials found.")

# -----------------------------------
# 2. Prepare Data for PyTorch
# -----------------------------------
X = np.concatenate([combined_standard, combined_oddball], axis=0)
y = np.concatenate([np.zeros(len(combined_standard)), np.ones(len(combined_oddball))], axis=0)

X = X.astype(np.float32)
X = X[:, np.newaxis, :, :]  # shape: (n_trials, 1, n_channels, n_times)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

import torch
from torch.utils.data import TensorDataset, DataLoader

X_train_torch = torch.tensor(X_train)
X_test_torch = torch.tensor(X_test)
y_train_torch = torch.tensor(y_train).float().unsqueeze(1)
y_test_torch = torch.tensor(y_test).float().unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_torch, y_train_torch), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=64)

# -----------------------------------
# 3. Define LMDA Model
# -----------------------------------
import torch.nn as nn
import torch.nn.functional as F

class LMDA(nn.Module):
    def __init__(self, chans=22, samples=1125, num_classes=1, depth=9, kernel=75, 
                 channel_depth1=24, channel_depth2=24):
        super(LMDA, self).__init__()

        self.channel_weight = nn.Parameter(torch.randn(depth, 1, chans), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)

        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, (1, kernel), padding=(0, kernel//2), bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
            nn.AvgPool2d((1, 4))
        )

        self.chanel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, (chans, 1), groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
            nn.AvgPool2d((1, 8))
        )

        with torch.no_grad():
            dummy = torch.ones(1, depth, chans, samples)
            out = torch.einsum('bdcw,hdc->bhcw', dummy, self.channel_weight)
            out = self.time_conv(out)
            out = self.chanel_conv(out)
            self.fc_input = out.numel()

        self.classifier = nn.Linear(self.fc_input, num_classes)

    def EEGDepthAttention(self, x):
        N, C, H, W = x.size()
        k = 7
        adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k//2, 0), bias=True).to(x.device)
        softmax = nn.Softmax(dim=-2)
        x_pool = adaptive_pool(x)
        x_transpose = x_pool.transpose(-2, -3)
        y = conv(x_transpose)
        y = softmax(y)
        y = y.transpose(-2, -3)
        return y * C * x

    def forward(self, x):
        x = torch.einsum('bdcw,hdc->bhcw', x, self.channel_weight)
        x = self.time_conv(x)
        x = self.EEGDepthAttention(x)
        x = self.chanel_conv(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.classifier(x))

# -----------------------------------
# 4. Train Model
# -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = LMDA(chans=X_train.shape[2], samples=X_train.shape[3]).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = (model(xb) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    print(f"Epoch {epoch+1}/20 - Test Acc: {correct / total:.4f}")

# -----------------------------------
# 5. Evaluation: Confusion Matrix & ROC
# -----------------------------------
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        outputs = model(xb)
        preds = outputs.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(yb.numpy())

all_preds = np.array(all_preds).flatten()
all_labels = np.array(all_labels).flatten()

# Confusion matrix
binary_preds = (all_preds > 0.5).astype(int)
cm = confusion_matrix(all_labels, binary_preds)
ConfusionMatrixDisplay(cm, display_labels=["Standard", "Oddball"]).plot()
plt.title("Confusion Matrix")
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_preds)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
