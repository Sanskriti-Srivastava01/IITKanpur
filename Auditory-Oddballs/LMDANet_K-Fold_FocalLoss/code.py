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

import torch
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler

X_full = torch.tensor(X)
y_full = torch.tensor(y).float().unsqueeze(1)
full_dataset = TensorDataset(X_full, y_full)

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
# 4. Focal Loss Definition
# -----------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean() if self.reduction == 'mean' else F_loss.sum()

# -----------------------------------
# 5. K-Fold Cross-Validation Training (with Focal Loss)
# -----------------------------------
from sklearn.model_selection import StratifiedKFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Calculate alpha based on class distribution (inverse frequency)
alpha = len(combined_standard) / (len(combined_standard) + len(combined_oddball))
criterion = FocalLoss(alpha=alpha, gamma=2)

fold_accuracies = []
all_preds = []
all_labels = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold+1}/{k_folds} ---")
    
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    
    train_loader = DataLoader(full_dataset, batch_size=64, sampler=train_sampler)
    val_loader = DataLoader(full_dataset, batch_size=64, sampler=val_sampler)

    model = LMDA(chans=X.shape[2], samples=X.shape[3]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    for epoch in range(20):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
    
    # Validation
    model.eval()
    correct, total = 0, 0
    fold_preds = []
    fold_labels = []
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            preds = (outputs > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            fold_preds.extend(outputs.cpu().numpy().flatten())
            fold_labels.extend(yb.cpu().numpy().flatten())
    
    fold_acc = correct / total
    fold_accuracies.append(fold_acc)
    all_preds.extend(fold_preds)
    all_labels.extend(fold_labels)
    
    print(f"Fold {fold+1} Accuracy: {fold_acc:.4f}")

print(f"\nAverage Cross-Validation Accuracy: {np.mean(fold_accuracies):.4f} (±{np.std(fold_accuracies):.4f})")

# -----------------------------------
# 6. Evaluation: Confusion Matrix & ROC
# -----------------------------------
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

binary_preds = (all_preds > 0.5).astype(int)
cm = confusion_matrix(all_labels, binary_preds)
ConfusionMatrixDisplay(cm, display_labels=["Standard", "Oddball"]).plot()
plt.title("Confusion Matrix (All Folds)")
plt.show()

fpr, tpr, _ = roc_curve(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_preds)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (All Folds)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# -----------------------------------
# 6. Enhanced Evaluation with Fold-by-Fold Analysis
# -----------------------------------
from sklearn.metrics import accuracy_score

# Initialize arrays to store all predictions and labels in original order
all_preds_ordered = np.zeros(len(X))
all_labels_ordered = np.zeros(len(X))
fold_metrics = []

# Re-run cross-validation to store indices
print("\n=== Fold-by-Fold Detailed Analysis ===")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Store validation indices
    val_indices = val_idx
    
    # Get predictions and labels for this fold
    fold_preds = all_preds[val_indices]
    fold_labels = all_labels[val_indices]
    
    # Calculate metrics
    fold_acc = accuracy_score(fold_labels, (fold_preds > 0.5))
    fold_auc = roc_auc_score(fold_labels, fold_preds)
    fold_metrics.append((fold_acc, fold_auc))
    
    # Create separate figures for each visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion Matrix
    cm = confusion_matrix(fold_labels, (fold_preds > 0.5).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=["Standard", "Oddball"]).plot(ax=axes[0])
    axes[0].set_title(f"Fold {fold+1} Confusion Matrix\nAccuracy: {fold_acc:.2f}")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(fold_labels, fold_preds)
    axes[1].plot(fpr, tpr, label=f"AUC = {fold_auc:.2f}")
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title(f"Fold {fold+1} ROC Curve")
    axes[1].legend()
    axes[1].grid()
    
    plt.tight_layout()
    plt.show()
    plt.close()  # Prevent figure accumulation in memory

# Print performance summary (remaining code remains the same)
print("\n=== Cross-Validation Performance Summary ===")
print(f"{'Fold':<6} | {'Accuracy':<8} | {'AUC':<6}")
print("-----------------------------------")
for i, (acc, auc) in enumerate(fold_metrics):
    print(f"Fold {i+1:<3} | {acc:.4f}    | {auc:.4f}")
print("-----------------------------------")
print(f"Mean  | {np.mean([m[0] for m in fold_metrics]):.4f}    | {np.mean([m[1] for m in fold_metrics]):.4f}")
print(f"Std   | ±{np.std([m[0] for m in fold_metrics]):.4f}  | ±{np.std([m[1] for m in fold_metrics]):.4f}")
