import os
import mne
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score

processed_folder = 'processed_data'  # Change if needed
epoch_files = [f for f in os.listdir(processed_folder) if f.endswith('-epo.fif')]

all_standard = []
all_oddball = []

# Modify this section in your data loading code:

for epo_file in epoch_files:
    print(f"Processing {epo_file}")
    epochs = mne.read_epochs(os.path.join(processed_folder, epo_file), preload=True)
    
    # Convert to float32 immediately after loading
    epochs._data = epochs.get_data().astype(np.float32)  # Critical memory optimization
    
    standard_keys = [k for k in epochs.event_id if 'standard' in k.lower()]
    oddball_keys = [k for k in epochs.event_id if 'oddball' in k.lower() or 'target' in k.lower()]
    
    if not standard_keys or not oddball_keys:
        print(f"Could not find standard or oddball events in {epo_file}. Skipping.")
        continue
        
    epochs_standard = epochs[standard_keys]
    epochs_oddball = epochs[oddball_keys]
    
    # Append already converted data
    all_standard.append(epochs_standard.get_data())
    all_oddball.append(epochs_oddball.get_data())

if all_standard and all_oddball:
    combined_standard = np.concatenate(all_standard, axis=0)
    combined_oddball = np.concatenate(all_oddball, axis=0)
else:
    raise RuntimeError("No standard or oddball trials found.")

# Prepare data for PyTorch
X = np.concatenate([combined_standard, combined_oddball], axis=0)
y = np.concatenate([np.zeros(len(combined_standard)), np.ones(len(combined_oddball))], axis=0)
X = X.astype(np.float32)
X = X[:, np.newaxis, :, :]  # (n_trials, 1, n_channels, n_times)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

import torch
from torch.utils.data import TensorDataset, DataLoader

X_train_torch = torch.tensor(X_train)
X_test_torch = torch.tensor(X_test)
y_train_torch = torch.tensor(y_train).float().unsqueeze(1)
y_test_torch = torch.tensor(y_test).float().unsqueeze(1)

batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_torch, y_train_torch), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=batch_size)
class EEGNet(nn.Module):
    def __init__(self, n_channels=120, n_timepoints=64):
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
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# K-Fold Cross Validation
k_folds = 5  # or any number you prefer
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

# Store metrics for each fold
results = {'acc': [], 'auc': [], 'precision': [], 'recall': [], 'f1': []}
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
# Before the K-fold loop
fold_metrics = []
all_y_true = []
all_y_score = []
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold+1}/{k_folds} ---")

    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    y_true = []
    y_pred = []
    y_score = []
    # Convert to torch tensors
    X_train_torch = torch.tensor(X_train)
    y_train_torch = torch.tensor(y_train).float().unsqueeze(1)
    X_test_torch = torch.tensor(X_test)
    y_test_torch = torch.tensor(y_test).float().unsqueeze(1)
    all_y_true.append(y_true)
    all_y_score.append(y_score)
    # DataLoader
    batch_size = 64
    train_loader = DataLoader(TensorDataset(X_train_torch, y_train_torch), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=batch_size)

    # Model, optimizer, loss
    model = EEGNet(n_channels=X.shape[2], n_timepoints=X.shape[3]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 20
    for epoch in range(epochs):
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
    y_true = []
    y_pred = []
    y_score = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            outputs = model(xb)
            y_true.extend(yb.cpu().numpy())
            y_pred.extend((outputs.cpu().numpy() > 0.5).astype(int))
            y_score.extend(outputs.cpu().numpy())
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    y_score = np.array(y_score).flatten()
    fold_metrics.append({
        'y_true': y_true,
        'y_pred': y_pred,
        'y_score': y_score})
    # Metrics
    results['acc'].append(accuracy_score(y_true, y_pred))
    results['auc'].append(roc_auc_score(y_true, y_score))
    results['precision'].append(precision_score(y_true, y_pred))
    results['recall'].append(recall_score(y_true, y_pred))
    results['f1'].append(f1_score(y_true, y_pred))

    print(f"Fold {fold+1} - Acc: {results['acc'][-1]:.3f}, AUC: {results['auc'][-1]:.3f}, "
          f"Precision: {results['precision'][-1]:.3f}, Recall: {results['recall'][-1]:.3f}, F1: {results['f1'][-1]:.3f}")

# Print average results
print("\n=== K-Fold Results ===")
for metric in results:
    print(f"{metric.upper()}: {np.mean(results[metric]):.3f} ± {np.std(results[metric]):.3f}")
fold_indices = range(1, k_folds+1)
plt.figure(figsize=(10,5))
plt.plot(fold_indices, results['acc'], marker='o', label='Accuracy')
plt.plot(fold_indices, results['f1'], marker='s', label='F1 Score')
plt.plot(fold_indices, results['auc'], marker='^', label='AUC')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('Per-fold Metrics')
plt.legend()
plt.grid(True)
plt.show()
# Confusion matrix for the last fold
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
    # Interpolate TPR
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
