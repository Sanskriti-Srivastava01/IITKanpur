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

print(f"Frequency resolution: {freq_resolution:.4f} Hz, Freq bins: {freq_bins}")

# Load data
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
    all_y.append(0 if file.endswith('_stay.csv') else 1)

X = np.stack(all_X, axis=0)
y = np.array(all_y)
n_channels = X.shape[1]
n_freq = X.shape[2]

print(f"Data shape: {X.shape}, Stay: {np.sum(y == 0)}, Leave: {np.sum(y == 1)}")

# Class weights
class_counts = Counter(y)
total = len(y)
w_stay = total / (2.0 * class_counts[0])
w_leave = total / (2.0 * class_counts[1])
class_weights = torch.tensor([w_stay, w_leave], dtype=torch.float32)

# -----------------------------------
# 2. Graph Construction
# -----------------------------------
def make_graph(epoch, label, thresh=0.7):
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
# 3. GCN Model with Feature Extraction
# -----------------------------------
class GCNNet(nn.Module):
    def __init__(self, node_feat_dim, out_dim=2, hid=32):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hid)
        self.conv2 = GCNConv(hid, hid)
        self.pool = global_mean_pool
        self.fc = nn.Linear(hid, out_dim)
    
    def forward(self, data, return_features=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        layer1_feat = x.clone()
        x = F.relu(self.conv2(x, edge_index))
        layer2_feat = x.clone()
        x = self.pool(x, batch)
        out = self.fc(x)
        
        if return_features:
            return out, {'layer1': layer1_feat, 'layer2': layer2_feat, 'pooled': x}
        return out

# -----------------------------------
# 4. Occlusion-based Feature Importance (FIXED)
# -----------------------------------
def compute_occlusion_map(model, batch_data, device, model_type='gcn'):
    """Compute channel-wise feature importance using occlusion"""
    model.eval()
    
    # Ensure batch_data is a single Data object (not a list)
    if isinstance(batch_data, list):
        batch_data = batch_data[0]
    
    # Move to device
    batch_data = batch_data.to(device)
    
    # Get baseline prediction
    with torch.no_grad():
        baseline_out = model(batch_data)
        baseline_logit = baseline_out[0, 1].item()  # Class 1 logit
    
    occlusion_map = np.zeros(n_channels)
    
    # Occlude each channel
    with torch.no_grad():
        for ch in range(n_channels):
            batch_occ = batch_data.clone()
            # Zero out all nodes in this channel
            batch_occ.x[:, ch] = 0.0
            
            occ_out = model(batch_occ)
            occ_logit = occ_out[0, 1].item()
            occlusion_map[ch] = baseline_logit - occ_logit
    
    return occlusion_map

# -----------------------------------
# 5. K-Fold Training with Feature Extraction
# -----------------------------------
print("\n" + "="*70)
print("K-FOLD CROSS-VALIDATION WITH FEATURE EXTRACTION")
print("="*70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_list, f1_list, cm_list, all_roc = [], [], [], []
all_fold_probs, all_fold_labels = [], []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Store features for visualization
layer1_features, layer2_features = [], []
occlusion_scores = []

for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold+1}/5")
    print("-" * 70)
    
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xtr2d = Xtr.reshape(len(Xtr), -1)
    ros = RandomOverSampler(random_state=42)
    Xbal, ybal = ros.fit_resample(Xtr2d, ytr)
    Xbal = Xbal.reshape(-1, n_channels, n_freq)
    
    # Create graphs
    train_graphs = [make_graph(Xbal[k], ybal[k]) for k in range(len(Xbal))]
    test_graphs = [make_graph(X[te_idx[k]], y[te_idx[k]]) for k in range(len(te_idx))]
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)

    # Initialize model
    model_gcn = GCNNet(node_feat_dim=n_freq, out_dim=2, hid=32).to(device)
    optimizer = torch.optim.Adam(model_gcn.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Training
    print("Training GCN...")
    for epoch in range(20):
        model_gcn.train()
        for batch in train_loader:
            batch = batch.to(device)
            out = model_gcn(batch)
            loss = criterion(out, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluation + Feature Extraction
    print("Evaluating and extracting features...")
    model_gcn.eval()
    y_true, y_pred, y_probs = [], [], []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            logits, feats = model_gcn(batch, return_features=True)
            y_probs.extend(F.softmax(logits, 1)[:, 1].cpu().numpy())
            y_pred.extend(torch.argmax(logits, 1).cpu().numpy())
            y_true.extend(batch.y.cpu().numpy())
            
            # Extract features from first batch (for visualization)
            if batch_idx == 0:
                # Layer 1 features (mean across batch and nodes)
                layer1_feat = feats['layer1'].mean(dim=0).cpu().numpy()
                layer1_features.append(layer1_feat)
                
                # Layer 2 features
                layer2_feat = feats['layer2'].mean(dim=0).cpu().numpy()
                layer2_features.append(layer2_feat)
                
                # Compute occlusion map for first sample in batch
                occ_map = compute_occlusion_map(model_gcn, batch[0:1], device, 'gcn')
                occlusion_scores.append(occ_map)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    auc = roc_auc_score(y_true, y_probs)
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    auc_list.append(auc)
    f1_list.append(f1)
    cm_list.append(cm)
    
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    all_roc.append((fpr, tpr))
    all_fold_probs.append(y_probs)
    all_fold_labels.append(y_true)
    
    print(f"  AUROC: {auc:.4f}, F1: {f1:.4f}")

# -----------------------------------
# 6. Visualize Layer Features (Figure 3 style)
# -----------------------------------
print("\n" + "="*70)
print("VISUALIZING LAYER-WISE FEATURES")
print("="*70)

if layer1_features:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Layer 1 heatmap
    layer1_stack = np.stack(layer1_features)
    im1 = axes[0].imshow(layer1_stack, cmap='viridis', aspect='auto')
    axes[0].set_title('GCN Layer 1: Feature Activations')
    axes[0].set_xlabel('Feature Dimension')
    axes[0].set_ylabel('Fold')
    plt.colorbar(im1, ax=axes[0])
    
    # Layer 2 heatmap
    layer2_stack = np.stack(layer2_features)
    im2 = axes[1].imshow(layer2_stack, cmap='viridis', aspect='auto')
    axes[1].set_title('GCN Layer 2: Feature Activations')
    axes[1].set_xlabel('Feature Dimension')
    axes[1].set_ylabel('Fold')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('gcn_layer_features.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: gcn_layer_features.png")

# -----------------------------------
# 7. Visualize Occlusion Maps (Figure 4 style)
# -----------------------------------
if occlusion_scores:
    fig, ax = plt.subplots(figsize=(12, 5))
    
    occ_mean = np.mean(occlusion_scores, axis=0)
    occ_std = np.std(occlusion_scores, axis=0)
    
    # Bar plot with error bars
    x_pos = np.arange(n_channels)
    ax.bar(x_pos, occ_mean, yerr=occ_std, capsize=5, alpha=0.7, color='steelblue')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(eeg_channels, rotation=45, ha='right')
    ax.set_ylabel('Feature Importance (Occlusion Score)')
    ax.set_xlabel('EEG Channels')
    ax.set_title('GCN: Channel-wise Feature Importance (via Occlusion)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('gcn_occlusion_map.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: gcn_occlusion_map.png")

# -----------------------------------
# 8. Final Results Summary
# -----------------------------------
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"AUROC:       {np.mean(auc_list):.3f} ± {np.std(auc_list):.3f}")
print(f"Weighted F1: {np.mean(f1_list):.3f} ± {np.std(f1_list):.3f}")

# Confusion matrix
cm_mean = np.mean(cm_list, axis=0)
fig_cm = ConfusionMatrixDisplay(cm_mean, display_labels=["Stay", "Leave"])
fig_cm.plot(cmap='Blues')
plt.title("Average Normalized Confusion Matrix (Stay/Leave)")
plt.tight_layout()
plt.savefig('confusion_matrix_weighted.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: confusion_matrix_weighted.png")

# ROC Curve
mean_fpr = np.linspace(0, 1, 100)
interp_tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in all_roc]
plt.figure(figsize=(7, 6))
plt.plot(mean_fpr, np.mean(interp_tprs, axis=0), label=f"Mean ROC, AUROC={np.mean(auc_list):.3f}", linewidth=2)
plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Mean ROC Curve across 5 folds (GCN)")
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_weighted.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: roc_curve_weighted.png")

# Precision-Recall
y_all = np.concatenate(all_fold_labels)
p_all = np.concatenate(all_fold_probs)
precision, recall, _ = precision_recall_curve(y_all, p_all, pos_label=1)
ap = average_precision_score(y_all, p_all, pos_label=1)

plt.figure(figsize=(7, 6))
plt.plot(recall, precision, label=f'Leave class (AP={ap:.3f})', linewidth=2, color='green')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve for Leave Detection")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('precision_recall_weighted.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: precision_recall_weighted.png")

print("\n" + "="*70)
print("✓ ALL VISUALIZATIONS AND RESULTS SAVED!")
print("="*70)

