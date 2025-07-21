import os
import mne
import numpy as np

processed_folder = 'processed_data'  # Change if needed
epoch_files = [f for f in os.listdir(processed_folder) if f.endswith('-epo.fif')]

all_standard = []
all_oddball = []

for epo_file in epoch_files:
    print(f"Processing {epo_file}")
    epochs = mne.read_epochs(os.path.join(processed_folder, epo_file), preload=True)
    
    # Identify event types
    standard_keys = [k for k in epochs.event_id if 'standard' in k.lower()]
    oddball_keys = [k for k in epochs.event_id if 'oddball' in k.lower() or 'target' in k.lower()]
    
    if not standard_keys or not oddball_keys:
        print(f"Could not find standard or oddball events in {epo_file}. Skipping.")
        continue
    
    # Extract trials for each
    epochs_standard = epochs[standard_keys]
    epochs_oddball = epochs[oddball_keys]
    
    # Append data as numpy arrays (n_trials, n_channels, n_times)
    all_standard.append(epochs_standard.get_data())
    all_oddball.append(epochs_oddball.get_data())

# Combine across all datasets
if all_standard and all_oddball:
    combined_standard = np.concatenate(all_standard, axis=0)
    combined_oddball = np.concatenate(all_oddball, axis=0)
    print(f"\nCombined NumPy array shapes:")
    print(f"  Standard: {combined_standard.shape}  # (n_trials, n_channels, n_times)")
    print(f"  Oddball:  {combined_oddball.shape}   # (n_trials, n_channels, n_times)")
else:
    print("No standard or oddball trials found in the datasets.")

import os
import mne
import numpy as np

processed_folder = 'processed_data'  # Change if needed
epoch_files = [f for f in os.listdir(processed_folder) if f.endswith('-epo.fif')]

all_standard = []
all_oddball = []

for epo_file in epoch_files:
    print(f"Processing {epo_file}")
    epochs = mne.read_epochs(os.path.join(processed_folder, epo_file), preload=True)
    
    # Identify event types
    standard_keys = [k for k in epochs.event_id if 'standard' in k.lower()]
    oddball_keys = [k for k in epochs.event_id if 'oddball' in k.lower() or 'target' in k.lower()]
    
    if not standard_keys or not oddball_keys:
        print(f"Could not find standard or oddball events in {epo_file}. Skipping.")
        continue
    
    # Extract trials for each
    epochs_standard = epochs[standard_keys]
    epochs_oddball = epochs[oddball_keys]
    
    # Append data as numpy arrays (n_trials, n_channels, n_times)
    all_standard.append(epochs_standard.get_data())
    all_oddball.append(epochs_oddball.get_data())

# Combine across all datasets
if all_standard and all_oddball:
    combined_standard = np.concatenate(all_standard, axis=0)
    combined_oddball = np.concatenate(all_oddball, axis=0)
    print(f"\nCombined NumPy array shapes:")
    print(f"  Standard: {combined_standard.shape}  # (n_trials, n_channels, n_times)")
    print(f"  Oddball:  {combined_oddball.shape}   # (n_trials, n_channels, n_times)")
    
    # Show a small part of the arrays
    print("\nFirst trial (standard):")
    print(combined_standard[0])  # shape: (n_channels, n_times)
    
    print("\nFirst trial (oddball):")
    print(combined_oddball[0])   # shape: (n_channels, n_times)
    
    # Optionally, show a summary (mean, std) for quick inspection
    print("\nStandard trials mean/std:", np.mean(combined_standard), np.std(combined_standard))
    print("Oddball trials mean/std:", np.mean(combined_oddball), np.std(combined_oddball))
else:
    print("No standard or oddball trials found in the datasets.")

print(combined_standard)
print(combined_oddball)

import numpy as np

# X: (n_trials, n_channels, n_times), y: (n_trials,)
X = np.concatenate([combined_standard, combined_oddball], axis=0)
y = np.concatenate([np.zeros(len(combined_standard)), np.ones(len(combined_oddball))], axis=0)

# Convert to float32 and add channel dimension for PyTorch (batch, 1, channels, time)
X = X.astype(np.float32)
X = X[:, np.newaxis, :, :]  # shape: (n_trials, 1, n_channels, n_times)

print("X shape:", X.shape)
print("y shape:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
import torch
from torch.utils.data import TensorDataset, DataLoader

X_train_torch = torch.tensor(X_train)
X_test_torch = torch.tensor(X_test)
# Labels must be float32 and shape (batch, 1)
y_train_torch = torch.tensor(y_train).float().unsqueeze(1)
y_test_torch = torch.tensor(y_test).float().unsqueeze(1)

batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_torch, y_train_torch), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_torch, y_test_torch), batch_size=batch_size)

import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, n_channels=120, n_timepoints=64):
        super(EEGNet, self).__init__()
        
        # Layer 1: Temporal convolution
        self.conv1 = nn.Conv2d(1, 16, (1, 63), padding=(0, 31))
        self.bn1 = nn.BatchNorm2d(16)
        
        # Layer 2: Spatial convolution (depthwise)
        self.conv2 = nn.Conv2d(16, 32, (n_channels, 1), groups=16)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((1, 4))
        
        # Layer 3: Separable convolution
        self.conv3 = nn.Conv2d(32, 32, (1, 16), padding=(0, 8), groups=32)
        self.conv4 = nn.Conv2d(32, 16, (1, 1))
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.AvgPool2d((1, 8))
        
        # Dynamic FC layer calculation
        self.fc_input_dim = self._get_fc_input_dim(n_channels, n_timepoints)
        self.fc = nn.Linear(self.fc_input_dim, 1)

    def _get_fc_input_dim(self, n_channels, n_timepoints):
        """Dynamically calculate FC layer input size."""
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

import torch.optim as optim
import torch

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = EEGNet(n_channels=X_train.shape[2], n_timepoints=X_train.shape[3]).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(20):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
    # Evaluation code

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
    acc = correct / total
    print(f"Epoch {epoch+1}/{20},Test Acc: {acc:.3f}")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:  # Replace `test_loader` with your DataLoader
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        preds = (torch.sigmoid(outputs) > 0.5).float()  # For binary classification
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
cmd = ConfusionMatrixDisplay(cm, display_labels=[0, 1])

# ROC curve
fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_preds)

# Plot
plt.figure(figsize=(10, 4))

# Confusion matrix
plt.subplot(1, 2, 1)
cmd.plot(cmap=plt.cm.Blues, ax=plt.gca(), colorbar=False)
plt.title('Confusion Matrix')

# ROC curve
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}', color='c')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()
