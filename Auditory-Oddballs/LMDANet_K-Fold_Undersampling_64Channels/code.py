# -----------------------------------
# 1. Load and Prepare 64-channel EEG Data
# -----------------------------------
import os
import mne
import numpy as np
import torch
import gc

processed_folder = 'processed_data'
epoch_files = [f for f in os.listdir(processed_folder) if f.endswith('-epo.fif')]

eeg_channels = [
    'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1',
    'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5',
    'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz',
    'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6',
    'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4',
    'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'
]

all_standard = []
all_oddball = []

for epo_file in epoch_files:
    print(f"Processing {epo_file}")
    epochs = mne.read_epochs(os.path.join(processed_folder, epo_file), preload=True)
    epochs.pick(eeg_channels)
    standard_keys = [k for k in epochs.event_id if 'standard' in k.lower()]
    oddball_keys = [k for k in epochs.event_id if 'oddball' in k.lower() or 'target' in k.lower()]
    if not standard_keys or not oddball_keys:
        print(f"Could not find standard or oddball events in {epo_file}. Skipping.")
        continue
    epochs_standard = epochs[standard_keys]
    epochs_oddball = epochs[oddball_keys]
    all_standard.append(epochs_standard.get_data().astype(np.float32))
    all_oddball.append(epochs_oddball.get_data().astype(np.float32))
    del epochs, epochs_standard, epochs_oddball
    gc.collect()

if all_standard and all_oddball:
    combined_standard = np.concatenate(all_standard, axis=0)
    combined_oddball = np.concatenate(all_oddball, axis=0)
    del all_standard, all_oddball
    gc.collect()
else:
    raise ValueError("No standard or oddball trials found.")

# -----------------------------------
# 1b. UNDERSAMPLE Majority Class (Standard)
# -----------------------------------
np.random.seed(42)
num_standard = len(combined_standard)
num_oddball = len(combined_oddball)
if num_standard > num_oddball:
    indices = np.random.choice(num_standard, num_oddball, replace=False)
    undersampled_standard = combined_standard[indices]
    print(f"Standard class undersampled from {num_standard} to {len(undersampled_standard)}")
    X = np.concatenate([undersampled_standard, combined_oddball], axis=0)
    y = np.concatenate([np.zeros(len(undersampled_standard)), np.ones(num_oddball)], axis=0)
elif num_oddball > num_standard:
    indices = np.random.choice(num_oddball, num_standard, replace=False)
    undersampled_oddball = combined_oddball[indices]
    print(f"Oddball class undersampled from {num_oddball} to {len(undersampled_oddball)}")
    X = np.concatenate([combined_standard, undersampled_oddball], axis=0)
    y = np.concatenate([np.zeros(num_standard), np.ones(len(undersampled_oddball))], axis=0)
else:
    X = np.concatenate([combined_standard, combined_oddball], axis=0)
    y = np.concatenate([np.zeros(num_standard), np.ones(num_oddball)], axis=0)

# Shuffle the balanced dataset
shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

print(f"Balanced dataset: {np.sum(y==0)} standard, {np.sum(y==1)} oddball")

X = X.astype(np.float32)
X = X[:, np.newaxis, :, :]  # shape: (n_trials, 1, 64, n_times)

from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler

X_full = torch.tensor(X)
y_full = torch.tensor(y).float().unsqueeze(1)
full_dataset = TensorDataset(X_full, y_full)

# -----------------------------------
# 2. Define LMDA Model for 64 Channels
# -----------------------------------
import torch.nn as nn
import torch.nn.functional as F

class LMDA(nn.Module):
    def __init__(self, chans=64, samples=None, num_classes=1, depth=9, kernel=75, 
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
        if samples is None:
            samples = X_full.shape[-1]
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
# 3. Training with Epoch-wise Logging
# -----------------------------------
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay

criterion = nn.BCELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:", device)

num_epochs = 10
k_folds = 5
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

fold_accuracies = []
all_preds = []
all_labels = []

# For plotting
train_losses_folds = []
val_losses_folds = []
train_accs_folds = []
val_accs_folds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold+1}/{k_folds} ---")
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(full_dataset, batch_size=64, sampler=train_sampler)
    val_loader = DataLoader(full_dataset, batch_size=64, sampler=val_sampler)
    model = LMDA(chans=64, samples=X.shape[3]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        train_losses.append(running_loss / total)
        train_accs.append(correct / total)

        # Validation
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item() * xb.size(0)
                preds = (outputs > 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_losses.append(val_loss / total)
        val_accs.append(correct / total)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")

    train_losses_folds.append(train_losses)
    val_losses_folds.append(val_losses)
    train_accs_folds.append(train_accs)
    val_accs_folds.append(val_accs)

    # Collect predictions for confusion matrix/ROC as before
    model.eval()
    fold_preds = []
    fold_labels = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            fold_preds.extend(outputs.cpu().numpy().flatten())
            fold_labels.extend(yb.cpu().numpy().flatten())
    all_preds.extend(fold_preds)
    all_labels.extend(fold_labels)
    fold_acc = np.mean(np.array(val_accs))
    fold_accuracies.append(fold_acc)
    print(f"Fold {fold+1} Mean Val Accuracy: {fold_acc:.4f}")

print(f"\nAverage Cross-Validation Accuracy: {np.mean(fold_accuracies):.4f} (±{np.std(fold_accuracies):.4f})")

# -----------------------------------
# 4. Plotting: Epoch-wise Loss and Accuracy for All Folds
# -----------------------------------
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(14, 6))
for i in range(k_folds):
    plt.plot(epochs, train_losses_folds[i], '--', label=f'Train Loss Fold {i+1}')
    plt.plot(epochs, val_losses_folds[i], '-', label=f'Val Loss Fold {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch-wise Loss per Fold')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(14, 6))
for i in range(k_folds):
    plt.plot(epochs, train_accs_folds[i], '--', label=f'Train Acc Fold {i+1}')
    plt.plot(epochs, val_accs_folds[i], '-', label=f'Val Acc Fold {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Epoch-wise Accuracy per Fold')
plt.legend()
plt.grid()
plt.show()

# -----------------------------------
# 5. Evaluation: Confusion Matrix & ROC (as before)
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
print("Final data shape:", X.shape)  # Should be (n_trials, 1, 64, n_times)
print("Number of EEG channels:", X.shape[2])  # Must be 64
