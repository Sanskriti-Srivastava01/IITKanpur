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
from sklearn.metrics import roc_auc_score, f1_score
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
from imblearn.over_sampling import RandomOverSampler
import networkx as nx

# ---------------------------
# 1. Load & preprocess data
# ---------------------------
input_folder = 'separated_classes'
eeg_channels = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'IO',
    'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6'
]
sampling_rate = 500

max_time = 0
file_list = []
for file in os.listdir(input_folder):
    if file.endswith(('_stay.csv', '_leave.csv')):
        path = os.path.join(input_folder, file)
        df = pd.read_csv(path)
        max_time = max(max_time, len(df))
        file_list.append(path)

def pad_trunc(data, max_len):
    if len(data) >= max_len:
        return data[:max_len]
    z = np.zeros((max_len - len(data), data.shape[1]))
    return np.vstack([data, z])

freq_resolution = sampling_rate / max_time
max_freq = 50
freq_bins = int(max_freq / freq_resolution) + 1

all_X, all_y, all_raw = [], [], []
for path in file_list:
    df = pd.read_csv(path)
    raw = df[eeg_channels].values
    raw_padded = pad_trunc(raw, max_time)

    fft = np.abs(np.fft.rfft(raw_padded.T, axis=1))
    freq_feat = fft[:, :freq_bins]

    if path.endswith('_stay.csv'):
        label = 0
    elif path.endswith('_leave.csv'):
        label = 1
    else:
        continue

    all_X.append(freq_feat)
    all_y.append(label)
    all_raw.append(raw_padded)

X = np.stack(all_X, axis=0)        # (N, C, F)
y = np.array(all_y)
all_raw = np.stack(all_raw, 0)     # (N, T, C)
n_channels, n_freq = X.shape[1], X.shape[2]
print("X shape:", X.shape, "y shape:", y.shape)

# ---------------------------
# 1b. Pre-training visualization
# ---------------------------
def plot_time_series_example(raw_data, label_name, sampling_rate, eeg_channels, save_path):
    t = np.arange(raw_data.shape[0]) / sampling_rate
    n_ch = raw_data.shape[1]
    plt.figure(figsize=(12, 8))
    offset = 0
    for i in range(n_ch):
        plt.plot(t, raw_data[:, i] + offset, color='k', linewidth=0.5)
        plt.text(t[0], offset, eeg_channels[i], fontsize=8, va='bottom')
        offset += np.std(raw_data[:, i]) * 5
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (stacked)')
    plt.title(f'Example raw EEG time series ({label_name})')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")

def build_corr_graph(epoch, thresh=0.7):
    corr = np.corrcoef(epoch)
    G = nx.Graph()
    for i, ch in enumerate(eeg_channels):
        G.add_node(i, label=ch)
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            if abs(corr[i, j]) > thresh:
                G.add_edge(i, j, weight=abs(corr[i, j]))
    return G

def plot_graph_example(epoch, label_name, thresh, save_path):
    G = build_corr_graph(epoch, thresh=thresh)
    plt.figure(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw(
        G, pos,
        with_labels=True,
        labels={i: eeg_channels[i] for i in G.nodes()},
        node_size=500, node_color='lightblue',
        width=weights, edge_color='gray'
    )
    plt.title(f'Correlation graph ({label_name}, thresh={thresh})')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")

stay_indices = np.where(y == 0)[0]
leave_indices = np.where(y == 1)[0]
if len(stay_indices) > 0 and len(leave_indices) > 0:
    ex_stay = stay_indices[0]
    ex_leave = leave_indices[0]
    plot_time_series_example(all_raw[ex_stay], 'stay', sampling_rate,
                             eeg_channels, 'example_stay_timeseries.png')
    plot_time_series_example(all_raw[ex_leave], 'leave', sampling_rate,
                             eeg_channels, 'example_leave_timeseries.png')
    plot_graph_example(X[ex_stay], 'stay', 0.7, 'example_stay_graph.png')
    plot_graph_example(X[ex_leave], 'leave', 0.7, 'example_leave_graph.png')
else:
    print("Not enough samples of both stay and leave to visualize.")

# ---------------------------
# 2. Class weights
# ---------------------------
class_counts = Counter(y)
total = len(y)
w_stay = total / (2.0 * class_counts[0])
w_leave = total / (2.0 * class_counts[1])
class_weights = torch.tensor([w_stay, w_leave], dtype=torch.float32)

# ---------------------------
# 3. Graph construction for GCN
# ---------------------------
def make_graph(epoch, label, thresh=0.7):
    node_feats = torch.tensor(epoch, dtype=torch.float32)   # (C, F)
    corr = np.corrcoef(epoch)                               # (C, C)
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

# ---------------------------
# 4. GCN model
# ---------------------------
class GCNNet(nn.Module):
    def __init__(self, node_feat_dim, hid=32, out_dim=2):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hid)
        self.conv2 = GCNConv(hid, hid)
        self.fc = nn.Linear(hid, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

# ---------------------------
# 5. Occlusion function
# ---------------------------
def channel_occlusion_importance(model, sample_data, device):
    model.eval()
    sample_data = sample_data.to(device)

    with torch.no_grad():
        logits = model(sample_data)
        true_class = sample_data.y.item()
        baseline_logit = logits[0, true_class].item()

    n_nodes = sample_data.x.size(0)
    importance = np.zeros(n_nodes)
    with torch.no_grad():
        for ch in range(n_nodes):
            occ_data = sample_data.clone()
            occ_data.x[ch, :] = 0.0
            logits_occ = model(occ_data)
            occ_logit = logits_occ[0, true_class].item()
            importance[ch] = baseline_logit - occ_logit
    return importance

# ---------------------------
# 6. K-fold training (100 epochs) + occlusion
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_list, f1_list = [], []
stay_importances, leave_importances = [], []

# Store examples for visualization
example_stay_importance = None
example_stay_raw = None
example_stay_freq = None
example_leave_importance = None
example_leave_raw = None
example_leave_freq = None

for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold+1}/5")
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xtr2d = Xtr.reshape(len(Xtr), -1)
    ros = RandomOverSampler(random_state=42)
    Xbal, ybal = ros.fit_resample(Xtr2d, ytr)
    Xbal = Xbal.reshape(-1, n_channels, n_freq)

    train_graphs = [make_graph(Xbal[i], ybal[i]) for i in range(len(Xbal))]
    test_graphs  = [make_graph(X[i], y[i]) for i in te_idx]

    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_graphs,  batch_size=16, shuffle=False)

    model = GCNNet(node_feat_dim=n_freq, hid=32, out_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    for epoch in range(100):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch+1) % 20 == 0:
            print(f"  Epoch {epoch+1}/100 - Loss: {running_loss/len(train_loader):.4f}")

    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch.y.cpu().numpy()

            y_true.extend(labels)
            y_pred.extend(preds)
            y_prob.extend(probs)

            for i in range(len(labels)):
                if preds[i] == labels[i]:
                    node_mask = (batch.batch == i)
                    x_i = batch.x[node_mask]

                    edge_mask = node_mask[batch.edge_index[0]] & node_mask[batch.edge_index[1]]
                    edge_index_i = batch.edge_index[:, edge_mask]
                    edge_attr_i = batch.edge_attr[edge_mask]

                    idxs = torch.nonzero(node_mask, as_tuple=False).view(-1)
                    old_to_new = {old.item(): new for new, old in enumerate(idxs)}
                    edge_index_i = edge_index_i.clone()
                    for k in range(edge_index_i.size(1)):
                        edge_index_i[0, k] = old_to_new[int(edge_index_i[0, k])]
                        edge_index_i[1, k] = old_to_new[int(edge_index_i[1, k])]

                    single = Data(
                        x=x_i,
                        edge_index=edge_index_i,
                        edge_attr=edge_attr_i,
                        y=torch.tensor(labels[i])
                    )
                    single.batch = torch.zeros(single.x.size(0), dtype=torch.long)

                    imp = channel_occlusion_importance(model, single, device)
                    
                    if labels[i] == 0:
                        stay_importances.append(imp)
                        # store one example
                        if example_stay_importance is None:
                            example_stay_importance = imp
                            example_stay_raw = all_raw[te_idx[i]]
                            example_stay_freq = X[te_idx[i]]
                    else:
                        leave_importances.append(imp)
                        # store one example
                        if example_leave_importance is None:
                            example_leave_importance = imp
                            example_leave_raw = all_raw[te_idx[i]]
                            example_leave_freq = X[te_idx[i]]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    auc = roc_auc_score(y_true, y_prob)
    f1  = f1_score(y_true, y_pred, average='weighted')
    auc_list.append(auc)
    f1_list.append(f1)
    print(f"  Fold {fold+1} AUROC: {auc:.4f}, Weighted F1: {f1:.4f}")

print("\nMean AUROC:", np.mean(auc_list), "±", np.std(auc_list))
print("Mean Weighted F1:", np.mean(f1_list), "±", np.std(f1_list))

# ---------------------------
# 7. Aggregate occlusion and plot (Figure 3-style: stay vs leave)
# ---------------------------
if len(stay_importances) > 0:
    stay_imp = np.mean(np.stack(stay_importances), axis=0)
else:
    stay_imp = np.zeros(n_channels)

if len(leave_importances) > 0:
    leave_imp = np.mean(np.stack(leave_importances), axis=0)
else:
    leave_imp = np.zeros(n_channels)

# Plot aggregate bar chart
x = np.arange(n_channels)
plt.figure(figsize=(12, 5))
plt.bar(x - 0.15, stay_imp, width=0.3, label='Stay', alpha=0.7)
plt.bar(x + 0.15, leave_imp, width=0.3, label='Leave', alpha=0.7)
plt.xticks(x, eeg_channels, rotation=45, ha='right')
plt.ylabel('Mean occlusion importance (Δ logit)')
plt.title('Channel-wise occlusion importance (GCN, correctly classified samples)')
plt.grid(axis='y', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('gcn_channel_occlusion_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: gcn_channel_occlusion_importance.png")

# ---------------------------
# 8. Figure 3-style: Occlusion maps overlaid on raw EEG + graph
# ---------------------------
def plot_occlusion_on_eeg_and_graph(raw_data, freq_data, importance, label_name, save_prefix):
    """
    Similar to Figure 3 in the paper:
    - Left: occlusion map overlaid on raw EEG time series
    - Right: occlusion values averaged over frequency + overlaid on graph
    """
    # Normalize importance to [0, 1]
    imp_norm = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: overlay on raw EEG
    ax = axes[0]
    t = np.arange(raw_data.shape[0]) / sampling_rate
    cmap = plt.cm.Reds
    norm = Normalize(vmin=0, vmax=1)
    
    offset = 0
    for i in range(n_channels):
        color = cmap(imp_norm[i])
        ax.plot(t, raw_data[:, i] + offset, color=color, linewidth=1.5, alpha=0.7)
        ax.text(t[0], offset, eeg_channels[i], fontsize=8, va='bottom')
        offset += np.std(raw_data[:, i]) * 5
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (stacked)')
    ax.set_title(f'Occlusion map on raw EEG ({label_name})')
    ax.grid(alpha=0.3)
    
    # Right panel: overlay on graph
    ax = axes[1]
    G = build_corr_graph(freq_data, thresh=0.7)
    pos = nx.spring_layout(G, seed=42)
    
    node_colors = [imp_norm[i] for i in G.nodes()]
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500,
                                    cmap=cmap, ax=ax, vmin=0, vmax=1)
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', ax=ax)
    nx.draw_networkx_labels(G, pos,
                           labels={i: eeg_channels[i] for i in G.nodes()},
                           font_size=8, ax=ax)
    
    ax.set_title(f'Occlusion map on graph ({label_name})')
    ax.axis('off')
    
    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Importance')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_occlusion_map.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_prefix}_occlusion_map.png")

if example_stay_importance is not None:
    plot_occlusion_on_eeg_and_graph(example_stay_raw, example_stay_freq,
                                     example_stay_importance, 'stay', 'example_stay')

if example_leave_importance is not None:
    plot_occlusion_on_eeg_and_graph(example_leave_raw, example_leave_freq,
                                     example_leave_importance, 'leave', 'example_leave')

# ---------------------------
# 9. Figure 7-style: Channel importance for classification
# ---------------------------
def plot_channel_importance_on_eeg(raw_data, importance, label_name, save_path):
    """
    Similar to Figure 7 (bottom panel): 
    Channel occlusion values replicated over time and overlaid on raw EEG
    """
    imp_norm = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    t = np.arange(raw_data.shape[0]) / sampling_rate
    cmap = plt.cm.RdYlBu_r
    norm = Normalize(vmin=0, vmax=1)
    
    offset = 0
    for i in range(n_channels):
        color = cmap(imp_norm[i])
        ax.plot(t, raw_data[:, i] + offset, color=color, linewidth=1, alpha=0.8)
        ax.text(t[0], offset, eeg_channels[i], fontsize=9, va='bottom', fontweight='bold')
        offset += np.std(raw_data[:, i]) * 5
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (stacked)')
    ax.set_title(f'Channel-wise occlusion importance ({label_name})')
    ax.grid(alpha=0.3)
    
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label('Importance')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")

if example_stay_importance is not None:
    plot_channel_importance_on_eeg(example_stay_raw, example_stay_importance,
                                    'stay', 'example_stay_classification_importance.png')

if example_leave_importance is not None:
    plot_channel_importance_on_eeg(example_leave_raw, example_leave_importance,
                                    'leave', 'example_leave_classification_importance.png')
