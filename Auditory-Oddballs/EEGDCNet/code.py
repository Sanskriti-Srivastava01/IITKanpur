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
# 3. Define EEG-DCNet Model
# -----------------------------------
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention block"""
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EEGDCNet(nn.Module):
    def __init__(self, chans=22, samples=1125, num_classes=1,
                 temporal_filters=40, spatial_filters=40, 
                 dilation_rates=[1, 2, 4], se_reduction=8):
        super(EEGDCNet, self).__init__()
        
        # Temporal Convolution Block (fixed)
        self.temporal_conv = nn.ModuleList()
        for dilation in dilation_rates:
            padding = (75 // 2) * dilation
            self.temporal_conv.append(
                nn.Sequential(
                    nn.Conv2d(1, temporal_filters, (1, 75),  # Input channels=1
                             padding=(0, padding),
                             dilation=(1, dilation)),
                    nn.BatchNorm2d(temporal_filters),
                    nn.GELU()
                )
            )
        
        # Spatial Convolution (fixed input channels)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(temporal_filters * len(dilation_rates),  # Correct input channels
                     spatial_filters, (chans, 1)),
            nn.BatchNorm2d(spatial_filters),
            nn.GELU(),
            nn.AvgPool2d((1, 4))
        )
        
        # Attention Blocks
        self.se1 = SEBlock(spatial_filters, se_reduction)
        self.se2 = SEBlock(spatial_filters, se_reduction)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self._calculate_fc_input(chans, samples), num_classes),
            nn.Sigmoid()
        )

    def _calculate_fc_input(self, chans, samples):
        with torch.no_grad():
            x = torch.randn(1, 1, chans, samples)
            # Process each temporal convolution separately
            temporal_outputs = []
            for conv in self.temporal_conv:
                temporal_outputs.append(conv(x))
            x = torch.cat(temporal_outputs, dim=1)  # Concatenate along channel dimension
            x = self.spatial_conv(x)
            x = self.se1(x)
            x = self.se2(x)
            return x.view(1, -1).size(1)

    def forward(self, x):
        # Multi-scale temporal processing
        temporal_features = []
        for conv in self.temporal_conv:
            temporal_features.append(conv(x))
        x = torch.cat(temporal_features, dim=1)
        
        # Spatial filtering
        x = self.spatial_conv(x)
        
        # Attention mechanisms
        x = self.se1(x)
        x = self.se2(x)
        
        # Classification
        x = x.view(x.size(0), -1)
        return self.fc(x)

# -----------------------------------
# 4. Train Model
# -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = EEGDCNet(chans=X_train.shape[2], samples=X_train.shape[3]).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

for epoch in range(20):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
    scheduler.step()
    
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
