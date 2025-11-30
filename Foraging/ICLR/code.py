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
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
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

# Epoch extraction and FFT
all_X, all_y = [], []
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
n_channels, n_freq = X.shape[1], X.shape[2]

# -----------------------------------
# 2. Graph Construction for Each Epoch
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
    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
    return Data(x=node_feats, edge_index=edge_index, edge_attr=edge_weight, y=torch.tensor(label, dtype=torch.long))

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
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.pool(x, batch)
        return self.fc(x)

# -----------------------------------
# 4. K-Fold Training, Metrics, Publishing Results
# -----------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_list, f1_list, cm_list, all_roc = [], [], [], []

for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
    # Oversample train for balance
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xtr2d = Xtr.reshape(len(Xtr), -1)
    ros = RandomOverSampler(random_state=42)
    Xbal, ybal = ros.fit_resample(Xtr2d, ytr)
    Xbal = Xbal.reshape(-1, n_channels, n_freq)
    train_graphs = [make_graph(Xbal[k], ybal[k]) for k in range(len(Xbal))]
    test_graphs  = [make_graph(X[te_idx[k]], y[te_idx[k]]) for k in range(len(te_idx))]
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)

    model = GCNNet(node_feat_dim=n_freq, out_dim=2).to('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    device = next(model.parameters()).device

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

    auc = roc_auc_score(y_true, y_probs)
    f1  = f1_score(y_true, y_pred, average='weighted')
    cm  = confusion_matrix(y_true, y_pred, normalize='true')
    auc_list.append(auc)
    f1_list.append(f1)
    cm_list.append(cm)

    # ROC for this fold
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    all_roc.append((fpr, tpr))

# ------------ Report/Plots --------------------
# Table with mean/std
print(f"AUROC:       {np.mean(auc_list):.3f} ± {np.std(auc_list):.3f}")
print(f"Weighted F1: {np.mean(f1_list):.3f} ± {np.std(f1_list):.3f}")
print("| Model | AUROC | Weighted F1 |")
print("|-------|-------------|-------------|")
print(f"| GCN   | {np.mean(auc_list):.3f} ± {np.std(auc_list):.3f} | {np.mean(f1_list):.3f} ± {np.std(f1_list):.3f} |")

# Mean normalized confusion matrix
cm_mean = np.mean(cm_list, axis=0)
fig_cm = ConfusionMatrixDisplay(cm_mean, display_labels=["Stay", "Leave"])
fig_cm.plot(cmap='Blues')
plt.title("Average Normalized Confusion Matrix (Stay/Leave)")
plt.show()

# Mean ROC
mean_fpr = np.linspace(0, 1, 100)
interp_tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in all_roc]
plt.plot(mean_fpr, np.mean(interp_tprs, axis=0), label=f"Mean ROC, AUROC={np.mean(auc_list):.2f}")
plt.plot([0,1], [0,1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("Mean ROC Curve across 5 folds (GCN)")
plt.show()
