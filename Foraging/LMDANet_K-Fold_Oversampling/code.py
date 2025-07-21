import os
import numpy as np
import mne
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold

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
print(f"Stay shape: {X_stay.shape}, Leave shape: {X_leave.shape}")

y_stay = np.zeros(len(X_stay), dtype=np.float32)
y_leave = np.ones(len(X_leave), dtype=np.float32)

# --- 2. Oversample minority class ---
diff = abs(len(X_stay) - len(X_leave))
if len(X_stay) > len(X_leave):
    idx = np.random.choice(len(X_leave), diff, replace=True)
    X_leave = np.concatenate([X_leave, X_leave[idx]], axis=0)
    y_leave = np.concatenate([y_leave, y_leave[idx]], axis=0)
elif len(X_leave) > len(X_stay):
    idx = np.random.choice(len(X_stay), diff, replace=True)
    X_stay = np.concatenate([X_stay, X_stay[idx]], axis=0)
    y_stay = np.concatenate([y_stay, y_stay[idx]], axis=0)

print(f"Oversampled shapes: Stay={X_stay.shape}, Leave={X_leave.shape}")

# --- 3. Combine and shuffle ---
X = np.concatenate([X_stay, X_leave], axis=0)
y = np.concatenate([y_stay, y_leave], axis=0)
perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

# Add channel dimension if needed (n_trials, 1, n_chans, n_times)
if X.ndim == 3:
    X = X[:, np.newaxis, :, :]

print(f"Final dataset shape: {X.shape}, Class balance: {(np.sum(y==0), np.sum(y==1))}")

# --- 4. Prepare PyTorch dataset ---
tensor_X = torch.tensor(X)
tensor_y = torch.tensor(y).float().unsqueeze(1)
dataset = TensorDataset(tensor_X, tensor_y)

# --- 5. Define LMDA-Net (simplified version) ---
class LMDA(nn.Module):
    def __init__(self, chans, samples):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (chans, 15), stride=(1,2))
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.GELU()
        self.pool1 = nn.AdaptiveAvgPool2d((1, 16))
        self.fc = nn.Linear(16*16, 1)

    def forward(self, x):
        # x: (batch, 1, chans, times)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = x.flatten(1)
        x = self.fc(x)
        return torch.sigmoid(x)

# ... (previous code until cross-validation section)

# --- 6. Train/Test Split and Cross-Validation ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
batch_size = 32
epochs = 10

# Initialize metric storage
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
    
    # Per-fold metrics
    fold_train_losses = []
    fold_val_losses = []
    fold_train_accs = []
    fold_val_accs = []
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_acc, n_train = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            train_acc += ((out > 0.5) == yb).sum().item()
            n_train += xb.size(0)
        train_loss /= n_train
        train_acc /= n_train

        model.eval()
        val_loss, val_acc, n_val = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                val_acc += ((out > 0.5) == yb).sum().item()
                n_val += xb.size(0)
        val_loss /= n_val
        val_acc /= n_val
        
        # Store epoch metrics
        fold_train_losses.append(train_loss)
        fold_train_accs.append(train_acc)
        fold_val_losses.append(val_loss)
        fold_val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}: Train loss {train_loss:.4f}, acc {train_acc:.4f} | Val loss {val_loss:.4f}, acc {val_acc:.4f}")
    
    # Save fold metrics
    train_losses_folds.append(fold_train_losses)
    val_losses_folds.append(fold_val_losses)
    train_accs_folds.append(fold_train_accs)
    val_accs_folds.append(fold_val_accs)

# --- 7. Plot Results ---
import matplotlib.pyplot as plt

# Plot loss curves
plt.figure(figsize=(14, 6))
for i in range(len(train_losses_folds)):
    plt.plot(train_losses_folds[i], '--', label=f'Train Loss Fold {i+1}')
    plt.plot(val_losses_folds[i], '-', label=f'Val Loss Fold {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid()
plt.show()

# Plot accuracy curves
plt.figure(figsize=(14, 6))
for i in range(len(train_accs_folds)):
    plt.plot(train_accs_folds[i], '--', label=f'Train Acc Fold {i+1}')
    plt.plot(val_accs_folds[i], '-', label=f'Val Acc Fold {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
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

# Re-run the validation for each fold using the trained model from each fold
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nEvaluating Fold {fold+1}...")
    # You need to reload/retrain your model for each fold if not kept in a list
    # If you have a list of trained models per fold, use: model = models[fold]
    # Otherwise, retrain or reload the model here as in your training loop

    # For demonstration, let's assume you re-train the model as in your training loop
    # (If you have the trained model for each fold, just use it here)
    model = LMDA(X.shape[2], X.shape[3]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    train_loader = DataLoader(dataset, batch_size=32, sampler=SubsetRandomSampler(train_idx))
    val_loader = DataLoader(dataset, batch_size=32, sampler=SubsetRandomSampler(val_idx))
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
    # Now evaluate
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
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
    acc = accuracy_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fold_auc = auc(fpr, tpr)
    fold_aucs.append(fold_auc)
    fold_accs.append(acc)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Leave"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Fold {fold+1} Confusion Matrix (Acc: {acc:.3f})")
    plt.show()

    # --- ROC Curve ---
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {fold_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Fold {fold+1} ROC Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    print(f"Fold {fold+1} Accuracy: {acc:.3f}, AUC: {fold_auc:.3f}")

# --- Average metrics across folds ---
print(f"\nAverage accuracy across folds: {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")
print(f"Average AUC across folds: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")

# --- (Optional) Plot average ROC curve ---
plt.figure(figsize=(8, 6))
for i in range(len(fold_probs)):
    fpr, tpr, _ = roc_curve(fold_labels[i], fold_probs[i])
    plt.plot(fpr, tpr, label=f'Fold {i+1} (AUC={auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Folds')
plt.legend()
plt.grid()
plt.show()
