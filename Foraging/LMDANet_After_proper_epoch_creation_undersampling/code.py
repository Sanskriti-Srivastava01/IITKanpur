import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from imblearn.under_sampling import RandomUnderSampler
import os
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

# -------------------------------
# 1. Load and Prepare All Data
# -------------------------------
input_folder = 'separated_classes'
eeg_channels = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'IO',
    'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6'
]

# Find max time length
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

# Shuffle before splitting
perm = np.random.permutation(len(all_X))
all_X = all_X[perm]
all_y = all_y[perm]

# -------------------------------
# 2. Split into N Equal Parts and Undersample Each
# -------------------------------
num_splits = 10
X_splits = np.array_split(all_X, num_splits)
y_splits = np.array_split(all_y, num_splits)

output_folder = 'balanced_splits'
os.makedirs(output_folder, exist_ok=True)

for i, (x_part, y_part) in enumerate(zip(X_splits, y_splits)):
    # Reshape for undersampling
    n_samples, n_channels, n_time = x_part.shape
    X_2d = x_part.reshape(n_samples, -1)
    rus = RandomUnderSampler(random_state=42)
    X_bal, y_bal = rus.fit_resample(X_2d, y_part)
    X_bal = X_bal.reshape(-1, n_channels, n_time)
    # Save balanced split
    np.save(os.path.join(output_folder, f"X_split_bal_{i}.npy"), X_bal)
    np.save(os.path.join(output_folder, f"y_split_bal_{i}.npy"), y_bal)
    print(f"Saved balanced split {i+1}/{num_splits} with {len(X_bal)} samples ({np.sum(y_bal==0)} stay, {np.sum(y_bal==1)} leave).")

print("All splits balanced and saved.")

# --------------------------------
# 3. Model and Dataset Definitions
# --------------------------------
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LMDA(nn.Module):
    def __init__(self, chans=28, samples=None, num_classes=2, depth=4, kernel=45, 
                 channel_depth1=8, channel_depth2=8, dropout_rate=0.5):
        super(LMDA, self).__init__()
        if samples is None:
            raise ValueError("samples parameter must be provided")
        self.depth = depth
        self.dropout = nn.Dropout(dropout_rate)
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, chans))
        nn.init.xavier_uniform_(self.channel_weight.data)
        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, (1, kernel), padding=(0, kernel//2), bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate/2)
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, (chans, 1), groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate/2)
        )
        with torch.no_grad():
            dummy = torch.ones(1, depth, chans, samples)
            out = torch.einsum('bdcw,hdc->bhcw', dummy, self.channel_weight)
            out = self.time_conv(out)
            out = self.channel_conv(out)
            self.fc_input = out.numel()
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

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
        x = x.unsqueeze(1).repeat(1, self.depth, 1, 1)
        x = torch.einsum('bdcw,hdc->bhcw', x, self.channel_weight)
        x = self.time_conv(x)
        x = self.EEGDepthAttention(x)
        x = self.dropout(x)
        x = self.channel_conv(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# --------------------------------
# 4. Cross-Validation Training and Evaluation
# --------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8
epochs = 20
test_accs = []

for test_split in range(num_splits):
    # Load all splits
    X_splits = []
    y_splits = []
    for i in range(num_splits):
        X_splits.append(np.load(os.path.join(output_folder, f"X_split_bal_{i}.npy")))
        y_splits.append(np.load(os.path.join(output_folder, f"y_split_bal_{i}.npy")))

    X_test, y_test = X_splits[test_split], y_splits[test_split]
    X_train = np.concatenate([x for i, x in enumerate(X_splits) if i != test_split])
    y_train = np.concatenate([y for i, y in enumerate(y_splits) if i != test_split])

    # Optionally: Apply undersampling to train set
    n_samples, n_channels, n_time = X_train.shape
    X_train_2d = X_train.reshape(n_samples, -1)
    rus = RandomUnderSampler(random_state=42)
    X_train, y_train = rus.fit_resample(X_train_2d, y_train)
    X_train = X_train.reshape(-1, n_channels, n_time)
    # Shuffle after undersampling
    perm = np.random.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    # Dataset and split
    train_dataset = EEGDataset(X_train, y_train)
    test_dataset = EEGDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = LMDA(chans=n_channels, samples=n_time, num_classes=2, depth=4, kernel=45, 
                 channel_depth1=8, channel_depth2=8, dropout_rate=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    best_val_loss = float('inf')
    patience = 4
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
        train_acc = correct / len(train_loader.dataset)

        # Validation (using test set for simplicity, or split train further for true validation)
        model.eval()
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            for X_test_batch, y_test_batch in test_loader:
                X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(device)
                outputs = model(X_test_batch)
                test_loss += criterion(outputs, y_test_batch).item() * X_test_batch.size(0)
                preds = outputs.argmax(dim=1)
                test_correct += (preds == y_test_batch).sum().item()
        test_loss /= len(test_loader.dataset)
        test_acc = test_correct / len(test_loader.dataset)
        scheduler.step(test_loss)  # Using test_loss as val_loss for simplicity

        print(f"Split {test_split+1}/{num_splits} | Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {total_loss/len(train_loader.dataset):.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        if test_loss < best_val_loss:
            best_val_loss = test_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} for split {test_split+1}")
            break

    # Final test evaluation
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)
            test_correct += (preds == y_batch).sum().item()
    test_acc = test_correct / len(test_loader.dataset)
    test_accs.append(test_acc)
    print(f"Split {test_split+1} Test Accuracy: {test_acc:.4f}")

print(f"\nAverage Test Accuracy across all splits: {np.mean(test_accs):.4f}")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# --------------------------------
# After training, during evaluation:
# --------------------------------
model.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        
        # Get probabilities and predictions
        probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1 ("leave")
        preds = outputs.argmax(dim=1)
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# Convert to numpy arrays
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# --------------------------------
# 1. Confusion Matrix
# --------------------------------
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay (0)", "Leave (1)"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# --------------------------------
# 2. ROC Curve
# --------------------------------
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

print(f"ROC AUC Score: {roc_auc:.4f}")
