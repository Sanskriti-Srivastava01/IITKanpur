import os
import numpy as np
import mne
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# --- 1. Load and Prepare Data ---
epo_folder = 'processed_results'
stay_files = [f for f in os.listdir(epo_folder) if f.endswith('-stay-epo.fif')]
leave_files = [f for f in os.listdir(epo_folder) if f.endswith('-leave-epo.fif')]

all_stay = []
all_leave = []

print("Loading stay epochs...")
for f in stay_files:
    epochs = mne.read_epochs(os.path.join(epo_folder, f), preload=True)
    all_stay.append(epochs.get_data().astype(np.float32))
print(f"Loaded {len(all_stay)} stay files.")

print("Loading leave epochs...")
for f in leave_files:
    epochs = mne.read_epochs(os.path.join(epo_folder, f), preload=True)
    all_leave.append(epochs.get_data().astype(np.float32))
print(f"Loaded {len(all_leave)} leave files.")

if not all_stay or not all_leave:
    raise ValueError('No stay or leave epochs found!')

X_stay = np.concatenate(all_stay, axis=0)
X_leave = np.concatenate(all_leave, axis=0)
print(f"Original shapes - Stay: {X_stay.shape}, Leave: {X_leave.shape}")

y_stay = np.zeros(len(X_stay), dtype=np.float32)
y_leave = np.ones(len(X_leave), dtype=np.float32)

# --- 2. Undersample majority class ---
if len(X_stay) > len(X_leave):
    # Undersample stay to match leave
    idx = np.random.choice(len(X_stay), len(X_leave), replace=False)
    X_stay = X_stay[idx]
    y_stay = y_stay[idx]
elif len(X_leave) > len(X_stay):
    # Undersample leave to match stay
    idx = np.random.choice(len(X_leave), len(X_stay), replace=False)
    X_leave = X_leave[idx]
    y_leave = y_leave[idx]

print(f"Undersampled shapes - Stay: {X_stay.shape}, Leave: {X_leave.shape}")

# --- 3. Combine and shuffle ---
X = np.concatenate([X_stay, X_leave], axis=0)
y = np.concatenate([y_stay, y_leave], axis=0)
perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

# Add channel dimension if needed
if X.ndim == 3:
    X = X[:, np.newaxis, :, :]

print(f"Final dataset shape: {X.shape}, Class balance: {np.unique(y, return_counts=True)}")

# --- 4. Prepare PyTorch dataset ---
tensor_X = torch.tensor(X)
tensor_y = torch.tensor(y).float().unsqueeze(1)
dataset = TensorDataset(tensor_X, tensor_y)

# --- 5. Define LMDA-Net (same as before) ---
class LMDA(nn.Module):
    def __init__(self, chans, samples):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (chans, 15), stride=(1,2))
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.GELU()
        self.pool1 = nn.AdaptiveAvgPool2d((1, 16))
        self.fc = nn.Linear(16*16, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = x.flatten(1)
        return torch.sigmoid(self.fc(x))

# --- 6. Cross-validation training loop ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
batch_size = 32
epochs = 10

train_losses_folds = []
val_losses_folds = []
train_accs_folds = []
val_accs_folds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold+1}/5")
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx))
    
    model = LMDA(X.shape[2], X.shape[3]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    fold_train_losses = []
    fold_val_losses = []
    fold_train_accs = []
    fold_val_accs = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * xb.size(0)
            train_correct += ((outputs > 0.5).float() == yb).sum().item()
            train_total += yb.size(0)
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = criterion(outputs, yb)
                
                val_loss += loss.item() * xb.size(0)
                val_correct += ((outputs > 0.5).float() == yb).sum().item()
                val_total += yb.size(0)
        
        # Store metrics
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        fold_train_losses.append(train_loss)
        fold_train_accs.append(train_acc)
        fold_val_losses.append(val_loss)
        fold_val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    # Save fold metrics
    train_losses_folds.append(fold_train_losses)
    val_losses_folds.append(fold_val_losses)
    train_accs_folds.append(fold_train_accs)
    val_accs_folds.append(fold_val_accs)

# --- 7. Plot results ---
plt.figure(figsize=(14, 6))
for i in range(len(train_losses_folds)):
    plt.plot(train_losses_folds[i], '--', label=f'Train Fold {i+1}')
    plt.plot(val_losses_folds[i], '-', label=f'Val Fold {i+1}')
plt.title("Training/Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(14, 6))
for i in range(len(train_accs_folds)):
    plt.plot(train_accs_folds[i], '--', label=f'Train Fold {i+1}')
    plt.plot(val_accs_folds[i], '-', label=f'Val Fold {i+1}')
plt.title("Training/Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Initialize lists to store fold-wise results
fold_preds = []
fold_labels = []
fold_probs = []
fold_aucs = []
fold_accs = []

# Collect predictions and labels during validation
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    val_loader = DataLoader(dataset, batch_size=32, sampler=SubsetRandomSampler(val_idx))
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            outputs = model(xb).cpu().numpy().flatten()
            preds = (outputs > 0.5).astype(int)
            y_true.extend(yb.numpy().flatten())
            y_pred.extend(preds)
            y_prob.extend(outputs)
    
    fold_labels.append(np.array(y_true))
    fold_preds.append(np.array(y_pred))
    fold_probs.append(np.array(y_prob))

# Fold-by-fold analysis
for i in range(len(fold_labels)):
    print(f"\n=== Fold {i+1} Analysis ===")
    
    # Confusion Matrix
    cm = confusion_matrix(fold_labels[i], fold_preds[i])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Leave"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {i+1} Confusion Matrix")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(fold_labels[i], fold_probs[i])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Fold {i+1} ROC Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    # Metrics
    acc = accuracy_score(fold_labels[i], fold_preds[i])
    fold_accs.append(acc)
    fold_aucs.append(roc_auc)
    print(f"Accuracy: {acc:.3f}")
    print(f"AUC: {roc_auc:.3f}")

# Aggregate results
print("\n=== Final Summary ===")
print(f"Average Accuracy: {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")
print(f"Average AUC: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")

# Combined ROC plot
plt.figure(figsize=(8, 6))
for i in range(len(fold_probs)):
    fpr, tpr, _ = roc_curve(fold_labels[i], fold_probs[i])
    plt.plot(fpr, tpr, label=f'Fold {i+1} (AUC={auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined ROC Curves')
plt.legend()
plt.grid()
plt.show()
