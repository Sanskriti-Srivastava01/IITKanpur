import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt

# --------------------------------
# 1. Load and Pad/Truncate Data
# --------------------------------
input_folder = 'separated_classes'
output_folder = 'equal_splits'
os.makedirs(output_folder, exist_ok=True)

eeg_channels = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'IO',
    'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6'
]

# Find maximum time length across all epochs
max_time = 0
for file in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file)
    if file.endswith(('_stay.csv', '_leave.csv')):
        df = pd.read_csv(file_path)
        max_time = max(max_time, len(df))

def pad_or_truncate(data, max_length):
    if len(data) > max_length:
        return data[:max_length]
    elif len(data) < max_length:
        padding = np.zeros((max_length - len(data), data.shape[1]))
        return np.vstack([data, padding])
    else:
        return data

# Load all epochs and labels
all_X, all_y = [], []
for file in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file)
    if not os.path.isfile(file_path):
        continue
    df = pd.read_csv(file_path)
    data = df[eeg_channels].values
    data_padded = pad_or_truncate(data, max_time)
    if file.endswith('_stay.csv'):
        all_X.append(data_padded)
        all_y.append(0)
    elif file.endswith('_leave.csv'):
        all_X.append(data_padded)
        all_y.append(1)

all_X = np.stack(all_X, axis=0)  # (n_epochs, max_time, n_channels)
all_X = np.transpose(all_X, (0, 2, 1))  # (n_epochs, n_channels, max_time)
all_y = np.array(all_y)

print(f"Loaded {len(all_X)} epochs. Each epoch shape: {all_X[0].shape}")

# --------------------------------
# 2. Shuffle and Split into Equal Parts
# --------------------------------
num_splits = 10  # Change as needed
perm = np.random.permutation(len(all_X))
all_X = all_X[perm]
all_y = all_y[perm]

X_splits = np.array_split(all_X, num_splits)
y_splits = np.array_split(all_y, num_splits)

# Save each split in the output folder
for i, (x_part, y_part) in enumerate(zip(X_splits, y_splits)):
    np.save(os.path.join(output_folder, f"X_split_{i}.npy"), x_part)
    np.save(os.path.join(output_folder, f"y_split_{i}.npy"), y_part)
    print(f"Saved split {i+1}/{num_splits} with {len(x_part)} samples to '{output_folder}'.")

# --------------------------------
# 3. EEGNet Model Definition (1 output logit for BCE)
# --------------------------------
class EEGNet(nn.Module):
    def __init__(self, n_channels, n_times, n_classes=1, 
                 F1=8, D=2, F2=16, kernel_length=64, dropout=0.25):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool2 = nn.AvgPool2d((1, 4))
        self.drop2 = nn.Dropout(dropout)
        self.separableConv_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.separableConv_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool3 = nn.AvgPool2d((1, 8))
        self.drop3 = nn.Dropout(dropout)
        self._fc_input_dim = self._get_fc_input_dim(n_channels, n_times)
        self.fc = nn.Linear(self._fc_input_dim, n_classes)

    def _get_fc_input_dim(self, n_channels, n_times):
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_times)
            x = self.conv1(x)
            x = self.bn1(x)
            x = nn.functional.elu(x)
            x = self.depthwiseConv(x)
            x = self.bn2(x)
            x = nn.functional.elu(x)
            x = self.pool2(x)
            x = self.drop2(x)
            x = self.separableConv_depth(x)
            x = self.separableConv_point(x)
            x = self.bn3(x)
            x = nn.functional.elu(x)
            x = self.pool3(x)
            x = self.drop3(x)
            return x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.elu(x)
        x = self.depthwiseConv(x)
        x = self.bn2(x)
        x = nn.functional.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = self.separableConv_depth(x)
        x = self.separableConv_point(x)
        x = self.bn3(x)
        x = nn.functional.elu(x)
        x = self.pool3(x)
        x = self.drop3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)  # No sigmoid here!

# --------------------------------
# 4. Cross-Validation Training with Weighted BCE
# --------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16
epochs = 20
test_accs = []

for test_split in range(num_splits):
    # Load splits
    X_test = np.load(os.path.join(output_folder, f"X_split_{test_split}.npy"))
    y_test = np.load(os.path.join(output_folder, f"y_split_{test_split}.npy"))
    X_train = np.concatenate([np.load(os.path.join(output_folder, f"X_split_{i}.npy")) for i in range(num_splits) if i != test_split])
    y_train = np.concatenate([np.load(os.path.join(output_folder, f"y_split_{i}.npy")) for i in range(num_splits) if i != test_split])

    # Ensure shape (n_samples, 1, n_channels, n_times)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    if X_train.ndim == 3:
        X_train = X_train[:, np.newaxis, :, :]
        X_test = X_test[:, np.newaxis, :, :]

    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # Calculate class weights for BCE
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-6)], dtype=torch.float32).to(device)

    train_loader = DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1)),
        batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test).unsqueeze(1)),
        batch_size=batch_size)

    # Model
    model = EEGNet(n_channels=X_train.shape[2], n_times=X_train.shape[3], n_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
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
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(yb.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    acc = (all_preds == all_labels).mean()
    test_accs.append(acc)
    print(f"Fold {test_split+1}/{num_splits} Test Accuracy: {acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay (0)", "Leave (1)"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Fold {test_split+1}")
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {test_split+1}')
    plt.legend(loc="lower right")
    plt.show()

print(f"\nAverage Test Accuracy across all folds: {np.mean(test_accs):.4f}")
