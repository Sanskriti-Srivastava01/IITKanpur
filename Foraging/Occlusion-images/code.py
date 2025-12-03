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
from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix, 
                             roc_curve, ConfusionMatrixDisplay, precision_recall_curve, 
                             average_precision_score)
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
# 3. GCN Model
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
# 4. Occlusion Map with Salience Heatmap (Figure 3)
# -----------------------------------
def compute_occlusion_map(model, batch_data, device):
    """Compute channel-wise feature importance"""
    if isinstance(batch_data, list):
        batch_data = batch_data[0]
    batch_data = batch_data.to(device)
    
    model.eval()
    with torch.no_grad():
        baseline_out = model(batch_data)
        baseline_logit = baseline_out[0, 1].item()
    
    occlusion_map = np.zeros(n_channels)
    with torch.no_grad():
        for ch in range(n_channels):
            batch_occ = batch_data.clone()
            batch_occ.x[:, ch] = 0.0
            occ_out = model(batch_occ)
            occ_logit = occ_out[0, 1].item()
            occlusion_map[ch] = baseline_logit - occ_logit
    
    return occlusion_map

# -----------------------------------
# 5. Coverage & Localization Metrics (Figure 4)
# -----------------------------------
def compute_coverage_localization(occlusion_scores, true_labels):
    """
    Coverage: % of important channels activated for correct predictions
    Localization: concentration of importance on few channels
    """
    coverage_scores = []
    localization_scores = []
    
    for occ, true_label in zip(occlusion_scores, true_labels):
        # Normalize occlusion scores
        occ_norm = (occ - occ.min()) / (occ.max() - occ.min() + 1e-8)
        
        # Coverage: proportion of channels with importance > threshold
        threshold = 0.3
        coverage = np.mean(occ_norm > threshold)
        coverage_scores.append(coverage)
        
        # Localization: entropy-based (lower = more localized)
        # Using top-k channels
        top_k = max(1, int(0.3 * n_channels))  # Top 30% channels
        top_indices = np.argsort(occ_norm)[-top_k:]
        top_values = occ_norm[top_indices]
        
        # Entropy of top-k
        top_values = top_values / (top_values.sum() + 1e-8)
        localization = -np.sum(top_values * np.log(top_values + 1e-8))
        localization_scores.append(localization)
    
    return np.array(coverage_scores), np.array(localization_scores)

# -----------------------------------
# 6. K-Fold Training
# -----------------------------------
print("\n" + "="*70)
print("K-FOLD CROSS-VALIDATION WITH FIGURE 3 & 4 METRICS")
print("="*70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_occlusion_scores = {'stay': [], 'leave': []}
all_coverage = {'stay': [], 'leave': []}
all_localization = {'stay': [], 'leave': []}
all_preds = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold+1}/5")
    
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xtr2d = Xtr.reshape(len(Xtr), -1)
    ros = RandomOverSampler(random_state=42)
    Xbal, ybal = ros.fit_resample(Xtr2d, ytr)
    Xbal = Xbal.reshape(-1, n_channels, n_freq)
    
    train_graphs = [make_graph(Xbal[k], ybal[k]) for k in range(len(Xbal))]
    test_graphs = [make_graph(X[te_idx[k]], y[te_idx[k]]) for k in range(len(te_idx))]
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)

    model_gcn = GCNNet(node_feat_dim=n_freq, out_dim=2, hid=32).to(device)
    optimizer = torch.optim.Adam(model_gcn.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    for epoch in range(20):
        model_gcn.train()
        for batch in train_loader:
            batch = batch.to(device)
            out = model_gcn(batch)
            loss = criterion(out, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model_gcn.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            logits = model_gcn(batch)
            preds = torch.argmax(logits, 1).cpu().numpy()
            true = batch.y.cpu().numpy()
            
            # Compute occlusion for each sample
            for sample_idx in range(logits.shape[0]):
                sample_data = batch[sample_idx:sample_idx+1]
                occ_map = compute_occlusion_map(model_gcn, sample_data, device)
                
                sample_label = true[sample_idx]
                sample_pred = preds[sample_idx]
                
                # Only for correctly predicted samples
                if sample_pred == sample_label:
                    all_preds.append(sample_pred)
                    if sample_label == 0:
                        all_occlusion_scores['stay'].append(occ_map)
                    else:
                        all_occlusion_scores['leave'].append(occ_map)

# -----------------------------------
# 7. Compute Coverage & Localization
# -----------------------------------
print("\nComputing coverage and localization metrics...")

for class_label in ['stay', 'leave']:
    if all_occlusion_scores[class_label]:
        cov, loc = compute_coverage_localization(
            np.array(all_occlusion_scores[class_label]),
            np.ones(len(all_occlusion_scores[class_label]))
        )
        all_coverage[class_label] = cov
        all_localization[class_label] = loc

# -----------------------------------
# 8. FIGURE 3: Occlusion Maps with Salience Heatmaps
# -----------------------------------
print("\nGenerating Figure 3: Occlusion Maps...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Figure 3: Occlusion Maps for Stay/Leave Detection', fontsize=14, fontweight='bold')

# (a) Stay - Full temporal sequence
if all_occlusion_scores['stay']:
    occ_stay = np.array(all_occlusion_scores['stay'])
    ax = axes[0, 0]
    im = ax.imshow(occ_stay[:min(20, len(occ_stay))], cmap='RdYlBu_r', aspect='auto')
    ax.set_title('(a) Stay: Occlusion Maps (Focal)')
    ax.set_xlabel('Frequency Bins')
    ax.set_ylabel('Sample Index')
    plt.colorbar(im, ax=ax, label='Importance')

# (b) Leave - Full temporal sequence
if all_occlusion_scores['leave']:
    occ_leave = np.array(all_occlusion_scores['leave'])
    ax = axes[0, 1]
    im = ax.imshow(occ_leave[:min(20, len(occ_leave))], cmap='RdYlBu_r', aspect='auto')
    ax.set_title('(b) Leave: Occlusion Maps (Focal)')
    ax.set_xlabel('Frequency Bins')
    ax.set_ylabel('Sample Index')
    plt.colorbar(im, ax=ax, label='Importance')

# (c) Stay - Generalized (averaged)
if all_occlusion_scores['stay']:
    ax = axes[1, 0]
    occ_stay_mean = np.repeat(occ_stay.mean(axis=0, keepdims=True), 10, axis=0)
    im = ax.imshow(occ_stay_mean, cmap='RdYlBu_r', aspect='auto')
    ax.set_title('(c) Stay: Averaged Occlusion (Generalized)')
    ax.set_xlabel('Frequency Bins')
    ax.set_ylabel('Repeated')
    plt.colorbar(im, ax=ax, label='Avg Importance')

# (d) Leave - Generalized (averaged)
if all_occlusion_scores['leave']:
    ax = axes[1, 1]
    occ_leave_mean = np.repeat(occ_leave.mean(axis=0, keepdims=True), 10, axis=0)
    im = ax.imshow(occ_leave_mean, cmap='RdYlBu_r', aspect='auto')
    ax.set_title('(d) Leave: Averaged Occlusion (Generalized)')
    ax.set_xlabel('Frequency Bins')
    ax.set_ylabel('Repeated')
    plt.colorbar(im, ax=ax, label='Avg Importance')

plt.tight_layout()
plt.savefig('figure3_occlusion_maps.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: figure3_occlusion_maps.png")

# -----------------------------------
# 9. FIGURE 4: Coverage & Localization Distribution
# -----------------------------------
print("\nGenerating Figure 4: Coverage & Localization Metrics...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Figure 4: Coverage and Localization Distributions', fontsize=14, fontweight='bold')

# (a) Coverage - Focal Seizures
ax = axes[0, 0]
coverage_data = [all_coverage['stay'], all_coverage['leave']]
colors_coverage = ['#1f77b4', '#ff7f0e']
labels_coverage = ['Stay', 'Leave']
bp1 = ax.boxplot(coverage_data, labels=labels_coverage, patch_artist=True)
for patch, color in zip(bp1['boxes'], colors_coverage):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Coverage Score')
ax.set_title('(a) Coverage Distribution (Focal)')
ax.grid(True, alpha=0.3)

# (b) Coverage - Generalized Seizures
ax = axes[0, 1]
bp2 = ax.boxplot(coverage_data, labels=labels_coverage, patch_artist=True)
for patch, color in zip(bp2['boxes'], colors_coverage):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Coverage Score')
ax.set_title('(b) Coverage Distribution (Generalized)')
ax.grid(True, alpha=0.3)

# (c) Localization - Focal Seizures
ax = axes[1, 0]
localization_data = [all_localization['stay'], all_localization['leave']]
bp3 = ax.boxplot(localization_data, labels=labels_coverage, patch_artist=True)
for patch, color in zip(bp3['boxes'], colors_coverage):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Localization Score (Entropy)')
ax.set_title('(c) Localization Distribution (Focal)')
ax.grid(True, alpha=0.3)

# (d) Localization - Generalized Seizures
ax = axes[1, 1]
bp4 = ax.boxplot(localization_data, labels=labels_coverage, patch_artist=True)
for patch, color in zip(bp4['boxes'], colors_coverage):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Localization Score (Entropy)')
ax.set_title('(d) Localization Distribution (Generalized)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure4_coverage_localization.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: figure4_coverage_localization.png")

# -----------------------------------
# 10. Channel-wise Importance Heatmap (Figure 4e-f style)
# -----------------------------------
print("\nGenerating channel-wise heatmaps...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mean occlusion per channel for Stay
if all_occlusion_scores['stay']:
    occ_stay_channels = np.mean(np.array(all_occlusion_scores['stay']), axis=0)
    im1 = axes[0].bar(range(n_channels), occ_stay_channels, color='steelblue', alpha=0.7)
    axes[0].set_xticks(range(n_channels))
    axes[0].set_xticklabels(eeg_channels, rotation=45, ha='right', fontsize=9)
    axes[0].set_ylabel('Mean Occlusion Score')
    axes[0].set_title('(e) Channel Importance: Stay Detection')
    axes[0].grid(True, alpha=0.3, axis='y')

# Mean occlusion per channel for Leave
if all_occlusion_scores['leave']:
    occ_leave_channels = np.mean(np.array(all_occlusion_scores['leave']), axis=0)
    im2 = axes[1].bar(range(n_channels), occ_leave_channels, color='coral', alpha=0.7)
    axes[1].set_xticks(range(n_channels))
    axes[1].set_xticklabels(eeg_channels, rotation=45, ha='right', fontsize=9)
    axes[1].set_ylabel('Mean Occlusion Score')
    axes[1].set_title('(f) Channel Importance: Leave Detection')
    axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figure4_channel_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Saved: figure4_channel_importance.png")

# -----------------------------------
# 11. Summary Statistics
# -----------------------------------
# -----------------------------------
# 11. Summary Statistics (FIXED)
# -----------------------------------
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"\nStay Samples (Correct Predictions): {len(all_occlusion_scores['stay'])}")
print(f"Leave Samples (Correct Predictions): {len(all_occlusion_scores['leave'])}")

# Use length checks instead of 'if array' which is ambiguous
if isinstance(all_coverage.get('stay'), (list, np.ndarray)) and len(all_coverage['stay']) > 0:
    print(f"\nCoverage (Stay):        {np.mean(all_coverage['stay']):.3f} ± {np.std(all_coverage['stay']):.3f}")
else:
    print("\nCoverage (Stay):        N/A (no correct stay samples or metrics not computed)")

if isinstance(all_coverage.get('leave'), (list, np.ndarray)) and len(all_coverage['leave']) > 0:
    print(f"Coverage (Leave):       {np.mean(all_coverage['leave']):.3f} ± {np.std(all_coverage['leave']):.3f}")
else:
    print("Coverage (Leave):       N/A (no correct leave samples or metrics not computed)")

if isinstance(all_localization.get('stay'), (list, np.ndarray)) and len(all_localization['stay']) > 0:
    print(f"\nLocalization (Stay):    {np.mean(all_localization['stay']):.3f} ± {np.std(all_localization['stay']):.3f}")
else:
    print("\nLocalization (Stay):    N/A")

if isinstance(all_localization.get('leave'), (list, np.ndarray)) and len(all_localization['leave']) > 0:
    print(f"Localization (Leave):   {np.mean(all_localization['leave']):.3f} ± {np.std(all_localization['leave']):.3f}")
else:
    print("Localization (Leave):   N/A")

print("\n✓ ALL FIGURES GENERATED!")
print("  - figure3_occlusion_maps.png")
print("  - figure4_coverage_localization.png")
print("  - figure4_channel_importance.png")

