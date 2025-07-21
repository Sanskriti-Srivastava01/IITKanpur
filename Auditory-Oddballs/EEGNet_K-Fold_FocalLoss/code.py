import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, reduction='mean'):
        """
        alpha: balance parameter, float or tensor (for class weights)
        gamma: focusing parameter
        reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits, targets):
        """
        logits: raw model outputs (no sigmoid), shape [batch, 1]
        targets: binary labels, shape [batch, 1]
        """
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * BCE_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

import os
import mne
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ========== 1. Data Preparation ==========
processed_folder = 'processed_data'  # Change as needed
epoch_files = [f for f in os.listdir(processed_folder) if f.endswith('-epo.fif')]

all_standard = []
all_oddball = []

for epo_file in epoch_files:
    print(f"Processing {epo_file}")
    epochs = mne.read_epochs(os.path.join(processed_folder, epo_file), preload=True)
    epochs._data = epochs.get_data().astype(np.float32)
    standard_keys = [k for k in epochs.event_id if 'standard' in k.lower()]
    oddball_keys = [k for k in epochs.event_id if 'oddball' in k.lower() or 'target' in k.lower()]
    if not standard_keys or not oddball_keys:
        print(f"Could not find standard or oddball events in {epo_file}. Skipping.")
        continue
    epochs_standard = epochs[standard_keys]
    epochs_oddball = epochs[oddball_keys]
    all_standard.append(epochs_standard.get_data())
    all_oddball.append(epochs_oddball.get_data())
    del epochs, epochs_standard, epochs_oddball

if all_standard and all_oddball:
    combined_standard = np.concatenate(all_standard, axis=0).astype(np.float32)
    combined_oddball = np.concatenate(all_oddball, axis=0).astype(np.float32)
else:
    raise RuntimeError("No standard or oddball trials found.")

X = np.concatenate([combined_standard, combined_oddball], axis=0)
y = np.concatenate([np.zeros(len(combined_standard)), np.ones(len(combined_oddball))], axis=0)
X = X[:, np.newaxis, :, :]  # (n_trials, 1, n_channels, n_times)
print(f"Data shape: {X.shape}, Labels shape: {y.shape}")

# ========== 2. EEGNet Model ==========
class EEGNet(nn.Module):
    def __init__(self, n_channels=79, n_timepoints=257):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 63), padding=(0, 31))
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, (n_channels, 1), groups=16)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((1, 4))
        self.conv3 = nn.Conv2d(32, 32, (1, 16), padding=(0, 8), groups=32)
        self.conv4 = nn.Conv2d(32, 16, (1, 1))
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.AvgPool2d((1, 8))
        self.fc_input_dim = self._get_fc_input_dim(n_channels, n_timepoints)
        self.fc = nn.Linear(self.fc_input_dim, 1)

    def _get_fc_input_dim(self, n_channels, n_timepoints):
        dummy = torch.randn(1, 1, n_channels, n_timepoints)
        with torch.no_grad():
            x = F.elu(self.conv1(dummy))
            x = self.bn1(x)
            x = F.elu(self.conv2(x))
            x = self.bn2(x)
            x = self.pool2(x)
            x = F.elu(self.conv3(x))
            x = F.elu(self.conv4(x))
            x = self.bn3(x)
            x = self.pool3(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = self.bn1(x)
        x = F.elu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x  # No sigmoid
import numpy as np

def check_normalization_scaling(X):
    """
    Check normalization and scaling of EEG trialwise data.

    Parameters:
    -----------
    X : np.ndarray
        EEG data of shape (n_trials, 1, n_channels, n_times) or (n_trials, n_channels, n_times)

    Returns:
    --------
    None (prints the results)
    """
    # If data has a singleton channel dimension, squeeze it
    if X.ndim == 4 and X.shape[1] == 1:
        X = X.squeeze(1)  # Now shape is (n_trials, n_channels, n_times)

    # Compute mean and std per trial (across all channels and times)
    mean_per_trial = np.mean(X, axis=(1,2))  # shape (n_trials,)
    std_per_trial = np.std(X, axis=(1,2))    # shape (n_trials,)

    # Overall mean and std
    overall_mean = np.mean(mean_per_trial)
    overall_std = np.mean(std_per_trial)

    print("First 5 trials' means:", mean_per_trial[:5])
    print("First 5 trials' stds :", std_per_trial[:5])
    print("Overall mean across trials:", overall_mean)
    print("Overall std across trials :", overall_std)
    print("\nInterpretation:")
    print("- If overall mean is close to 0 and overall std is close to 1, your data is likely normalized and scaled.")
    print("- If not, consider applying normalization (e.g., z-scoring) before model input.")

# ========== 3. K-Fold Training, Focal Loss, and Visualization ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
k_folds = 5
batch_size = 64
epochs = 15

skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
results = {'acc': [], 'auc': [], 'precision': [], 'recall': [], 'f1': []}
fold_metrics = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold+1}/{k_folds} ---")
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Diagnostic: Print class balance
    print(f"Train class 0: {np.sum(y_train == 0)}, class 1: {np.sum(y_train == 1)} | "
          f"Test class 0: {np.sum(y_test == 0)}, class 1: {np.sum(y_test == 1)}")

    # DataLoader
    X_train_torch = torch.tensor(X_train)
    y_train_torch = torch.tensor(y_train).float().unsqueeze(1)
    X_test_torch = torch.tensor(X_test)
    y_test_torch = torch.tensor(y_test).float().unsqueeze(1)
    train_loader = DataLoader(TensorDataset(X_train_torch, y_train_torch), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=batch_size)

    # Model, optimizer, Focal loss
    model = EEGNet(n_channels=X.shape[2], n_timepoints=X.shape[3]).to(device)
    # Optionally, set alpha to the positive class frequency or 0.25
    alpha = float(np.sum(y_train == 1)) / len(y_train)
    criterion = FocalLoss(alpha=alpha, gamma=2.0, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    X = np.concatenate([combined_standard, combined_oddball], axis=0)
    X = X[:, np.newaxis, :, :]  # (n_trials, 1, n_channels, n_times)
    check_normalization_scaling(X)
    # Training
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            y_true.extend(yb.cpu().numpy())
            y_pred.extend((probs.cpu().numpy() > 0.5).astype(int))
            y_score.extend(probs.cpu().numpy())
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    y_score = np.array(y_score).flatten()

    # Diagnostic: Print prediction distribution
    print(f"True label counts: {np.bincount(y_true.astype(int))}, Predicted label counts: {np.bincount(y_pred.astype(int))}")

    results['acc'].append(accuracy_score(y_true, y_pred))
    results['auc'].append(roc_auc_score(y_true, y_score))
    results['precision'].append(precision_score(y_true, y_pred, zero_division=0))
    results['recall'].append(recall_score(y_true, y_pred, zero_division=0))
    results['f1'].append(f1_score(y_true, y_pred, zero_division=0))
    fold_metrics.append({'y_true': y_true, 'y_pred': y_pred, 'y_score': y_score})

    print(f"Fold {fold+1} - Acc: {results['acc'][-1]:.3f}, AUC: {results['auc'][-1]:.3f}, "
          f"Precision: {results['precision'][-1]:.3f}, Recall: {results['recall'][-1]:.3f}, F1: {results['f1'][-1]:.3f}")

# ========== 4. Visualization ==========

plt.figure(figsize=(10,5))
fold_indices = range(1, k_folds+1)
plt.plot(fold_indices, results['acc'], marker='o', label='Accuracy')
plt.plot(fold_indices, results['f1'], marker='s', label='F1 Score')
plt.plot(fold_indices, results['auc'], marker='^', label='AUC')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('Per-fold Metrics')
plt.legend()
plt.grid(True)
plt.show()

last_y_true = fold_metrics[-1]['y_true']
last_y_pred = fold_metrics[-1]['y_pred']
cm = confusion_matrix(last_y_true, last_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Last Fold)')
plt.show()

plt.figure(figsize=(8,6))
mean_fpr = np.linspace(0, 1, 100)
tprs = []
aucs = []
for i, fm in enumerate(fold_metrics):
    fpr, tpr, _ = roc_curve(fm['y_true'], fm['y_score'])
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.5, label=f'Fold {i+1} AUC={roc_auc:.2f}')
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC={mean_auc:.2f})', lw=2)
plt.plot([0,1], [0,1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Across Folds')
plt.legend()
plt.grid(True)
plt.show()

print("\n=== K-Fold Results ===")
for metric in results:
    print(f"{metric.upper()}: {np.mean(results[metric]):.3f} ± {np.std(results[metric]):.3f}")
