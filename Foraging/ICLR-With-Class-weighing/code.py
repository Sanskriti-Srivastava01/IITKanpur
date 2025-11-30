import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, 
                             confusion_matrix, roc_curve, ConfusionMatrixDisplay,
                             precision_recall_curve, average_precision_score)
from collections import Counter
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

# ---------------------------
# 1. Load and Preprocess Data
# ---------------------------
input_folder = 'separated_classes'
eeg_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'IO',
                'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6']

sampling_rate = 500
max_time = 0

print("Finding max time length...")
for file in os.listdir(input_folder):
    if file.endswith(('_stay.csv', '_leave.csv')):
        df = pd.read_csv(os.path.join(input_folder, file))
        max_time = max(max_time, len(df))

def pad_trunc(data, max_len):
    if len(data) >= max_len:
        return data[:max_len]
    z = np.zeros((max_len - len(data), data.shape[1]))
    return np.vstack([data, z])

freq_resolution = sampling_rate / max_time
max_freq = 50
freq_bins = int(max_freq / freq_resolution) + 1

print(f"Frequency resolution: {freq_resolution:.4f} Hz")
print(f"Frequency bins: {freq_bins}")

# Epoch extraction and FFT
all_X, all_y = [], []
print("Loading EEG data...")
for file in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file)
    if not os.path.isfile(file_path):
        continue
    df = pd.read_csv(file_path)
    data = df[eeg_channels].values
    data = pad_trunc(data, max_time)
    fft = np.abs(np.fft.rfft(data.T, axis=1))
    freq_feat = fft[:, :freq_bins]
    all_X.append(freq_feat)
    if file.endswith('_stay.csv'):
        all_y.append(0)
    elif file.endswith('_leave.csv'):
        all_y.append(1)

X = np.stack(all_X, axis=0)
y = np.array(all_y)
n_channels = X.shape[1]
n_freq = X.shape[2]

print(f"Data shape: {X.shape}")
print(f"Stay samples: {np.sum(y == 0)}, Leave samples: {np.sum(y == 1)}")

# -----------------------------------
# 2. Compute Class Weights (before K-fold)
# -----------------------------------
class_counts = Counter(y)
total = len(y)
w_stay = total / (2.0 * class_counts[0])
w_leave = total / (2.0 * class_counts[1])
class_weights = torch.tensor([w_stay, w_leave], dtype=torch.float32)
print(f"Class weights - Stay: {w_stay:.3f}, Leave: {w_leave:.3f}")

# -----------------------------------
# 3. Graph Construction for Each Epoch
# -----------------------------------
def make_graph(epoch, label, thresh=0.7):
    """Create graph from correlation matrix"""
    node_feats = torch.tensor(epoch, dtype=torch.float32)
    corr = np.corrcoef(epoch)
    edge_index, edge_weight = [], []
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            if abs(corr[i, j]) > thresh:
                edge_index += [[i, j], [j, i]]
                edge_weight += [abs(corr[i,j]), abs(corr[i,j])]
    if not edge_index:
        edge_index = [[i, i] for i in range(n_channels)]
        edge_weight = [1.0] * n_channels
    edge_index = torch.tensor(edge_index, dtype=torch.long).T.contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
    return Data(x=node_feats, edge_index=edge_index, edge_attr=edge_weight, 
                y=torch.tensor(label, dtype=torch.long))

# -----------------------------------
# 4. GCN Model
# -----------------------------------
class GCNNet(nn.Module):
    def __init__(self, node_feat_dim, out_dim=2, hid=32):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hid)
        self.conv2 = GCNConv(hid, hid)
        self.pool = global_mean_pool
        self.fc = nn.Linear(hid, out_dim)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.pool(x, batch)
        return self.fc(x)

# -----------------------------------
# 5. K-Fold Training with Class Weights + Metrics
# -----------------------------------
print("\n" + "="*60)
print("K-FOLD CROSS-VALIDATION WITH CLASS WEIGHTING")
print("="*60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_list, f1_list, cm_list, all_roc = [], [], [], []
all_fold_probs, all_fold_labels = [], []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold+1}/5")
    print("-" * 40)
    
    # Oversample train for balance
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xtr2d = Xtr.reshape(len(Xtr), -1)
    ros = RandomOverSampler(random_state=42)
    Xbal, ybal = ros.fit_resample(Xtr2d, ytr)
    Xbal = Xbal.reshape(-1, n_channels, n_freq)
    
    train_graphs = [make_graph(Xbal[k], ybal[k]) for k in range(len(Xbal))]
    test_graphs = [make_graph(X[te_idx[k]], y[te_idx[k]]) for k in range(len(te_idx))]
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)

    model = GCNNet(node_feat_dim=n_freq, out_dim=2, hid=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # CLASS-WEIGHTED LOSS
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Training loop
    for epoch in range(20):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # --- Evaluation ---
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch)
            y_probs.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
            y_pred.extend(torch.argmax(logits, 1).cpu().numpy())
            y_true.extend(batch.y.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    auc = roc_auc_score(y_true, y_probs)
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    auc_list.append(auc)
    f1_list.append(f1)
    cm_list.append(cm)

    # ROC for this fold
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    all_roc.append((fpr, tpr))
    
    # Store for precision-recall analysis
    all_fold_probs.append(y_probs)
    all_fold_labels.append(y_true)
    
    print(f"AUROC: {auc:.4f}, Weighted F1: {f1:.4f}")

# ------------ FINAL RESULTS AND PLOTS --------------------
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"AUROC:       {np.mean(auc_list):.3f} ± {np.std(auc_list):.3f}")
print(f"Weighted F1: {np.mean(f1_list):.3f} ± {np.std(f1_list):.3f}")

print("\n| Model | AUROC | Weighted F1 |")
print("|-------|-------|-------------|")
print(f"| GCN   | {np.mean(auc_list):.3f} ± {np.std(auc_list):.3f} | {np.mean(f1_list):.3f} ± {np.std(f1_list):.3f} |")

# --- 1. Mean Normalized Confusion Matrix (FIXED) ---
cm_mean = np.mean(cm_list, axis=0)
fig_cm = ConfusionMatrixDisplay(cm_mean, display_labels=["Stay", "Leave"])
fig_cm.plot(cmap='Blues')
plt.title("Average Normalized Confusion Matrix (Stay/Leave)")
plt.tight_layout()
plt.savefig('confusion_matrix_weighted.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 2. Mean ROC ---
mean_fpr = np.linspace(0, 1, 100)
interp_tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in all_roc]
plt.figure(figsize=(7, 6))
plt.plot(mean_fpr, np.mean(interp_tprs, axis=0), label=f"Mean ROC, AUROC={np.mean(auc_list):.3f}", linewidth=2)
plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Mean ROC Curve across 5 folds (GCN with Class Weighting)")
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_weighted.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 3. Precision-Recall Curve for Leave Class ---
y_all = np.concatenate(all_fold_labels)
p_all = np.concatenate(all_fold_probs)

precision, recall, thresholds = precision_recall_curve(y_all, p_all, pos_label=1)
ap = average_precision_score(y_all, p_all, pos_label=1)

plt.figure(figsize=(7, 6))
plt.plot(recall, precision, label=f'Leave class (AP={ap:.3f})', linewidth=2, color='green')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve for Leave Detection (GCN with Class Weighting)")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('precision_recall_weighted.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 4. Find Best Threshold for Leave ---
best_thresh = 0.5
best_f1_leave = 0.0
thresholds_tested = []
f1_leave_list = []

for t in np.linspace(0.1, 0.9, 17):
    preds_t = (p_all >= t).astype(int)
    f1_leave = f1_score(y_all, preds_t, pos_label=1, zero_division=0)
    thresholds_tested.append(t)
    f1_leave_list.append(f1_leave)
    if f1_leave > best_f1_leave:
        best_f1_leave = f1_leave
        best_thresh = t

print(f"\nBest threshold for Leave: {best_thresh:.2f}")
print(f"F1_leave at best threshold: {best_f1_leave:.4f}")

# Plot threshold tuning
plt.figure(figsize=(7, 5))
plt.plot(thresholds_tested, f1_leave_list, 'o-', linewidth=2, markersize=6, color='purple')
plt.axvline(x=best_thresh, color='red', linestyle='--', label=f'Best threshold: {best_thresh:.2f}')
plt.xlabel("Classification Threshold")
plt.ylabel("F1-Score (Leave Class)")
plt.title("Threshold Tuning for Leave Detection")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('threshold_tuning.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 5. Per-class Metrics at Best Threshold ---
preds_best = (p_all >= best_thresh).astype(int)
cm_best = confusion_matrix(y_all, preds_best)
cm_best_norm = cm_best.astype('float') / cm_best.sum(axis=1)[:, np.newaxis]

tn, fp, fn, tp = cm_best.ravel()
sensitivity_leave = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity_leave = tn / (tn + fp) if (tn + fp) > 0 else 0
precision_leave = tp / (tp + fp) if (tp + fp) > 0 else 0

print(f"\nPer-class metrics at threshold {best_thresh:.2f}:")
print(f"  Leave Sensitivity (Recall): {sensitivity_leave:.4f}")
print(f"  Leave Specificity: {specificity_leave:.4f}")
print(f"  Leave Precision: {precision_leave:.4f}")
print(f"  Leave F1-Score: {best_f1_leave:.4f}")

# --- 6. Summary Statistics Table ---
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Mean AUROC:       {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
print(f"Mean Weighted F1: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
print(f"Mean Confusion Matrix (row-normalized):")
print(cm_mean)
print(f"\nAverage Precision (Leave): {ap:.4f}")
print(f"Best Threshold: {best_thresh:.2f}")
print(f"Leave F1 @ Best Threshold: {best_f1_leave:.4f}")

print("\n✓ All results saved as PNG files!")
