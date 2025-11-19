import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data as GeoData
from torch_geometric.loader import DataLoader  # <-- CORRECTED: Use torch_geometric DataLoader
from torch_geometric.nn import ChebConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

# ================================================================
# 1. DATA LOADING
# ================================================================
input_folder = 'separated_classes'
eeg_channels = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'IO',
    'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6'
]

sampling_rate = 500
max_time = 0

print("Finding max time length...")
for file in os.listdir(input_folder):
    if file.endswith(('_stay.csv', '_leave.csv')):
        df = pd.read_csv(os.path.join(input_folder, file))
        max_time = max(max_time, len(df))

def pad_or_truncate(data, max_length):
    if len(data) > max_length:
        return data[:max_length]
    elif len(data) < max_length:
        padding = np.zeros((max_length - len(data), data.shape[1]))
        return np.vstack([data, padding])
    else:
        return data

freq_resolution = sampling_rate / max_time
max_freq = 50
freq_bins = int(max_freq / freq_resolution) + 1

print(f"Frequency resolution: {freq_resolution:.4f} Hz")
print(f"Frequency bins (0-{max_freq} Hz): {freq_bins}")
print(f"Max time samples: {max_time}")

all_X, all_y = [], []
print("\nLoading EEG data...")
for file in os.listdir(input_folder):
    file_path = os.path.join(input_folder, file)
    if not os.path.isfile(file_path):
        continue
    df = pd.read_csv(file_path)
    data = df[eeg_channels].values
    data_padded = pad_or_truncate(data, max_time)
    fft_vals = np.abs(np.fft.rfft(data_padded.T, axis=1))
    freq_feat = fft_vals[:, :freq_bins]
    
    if file.endswith('_stay.csv'):
        all_X.append(freq_feat)
        all_y.append(0)
    elif file.endswith('_leave.csv'):
        all_X.append(freq_feat)
        all_y.append(1)

X = np.stack(all_X, axis=0)
y = np.array(all_y)
n_channels = X.shape[1]
n_freq = X.shape[2]

print(f"Data shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Stay samples: {np.sum(y == 0)}, Leave samples: {np.sum(y == 1)}")

# ================================================================
# 2. PLOT GRAPH STRUCTURES FOR STAY AND LEAVE
# ================================================================
print("\n" + "="*60)
print("PLOTTING GRAPH STRUCTURES")
print("="*60)

def compute_correlation_matrix(epoch_data):
    """Compute correlation matrix for one epoch"""
    return np.corrcoef(epoch_data)

def build_graph(epoch_data, corr_thresh=0.7):
    """Build NetworkX graph from correlation matrix"""
    n_channels = epoch_data.shape[0]
    corr = compute_correlation_matrix(epoch_data)
    G = nx.Graph()
    G.add_nodes_from(range(n_channels))
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            if abs(corr[i, j]) > corr_thresh:
                G.add_edge(i, j, weight=abs(corr[i, j]))
    return G, corr

def plot_eeg_graph_and_corr_heatmap(epoch_data, label, channel_names, corr_thresh=0.7):
    """Plot both the graph structure and correlation heatmap"""
    G, corr = build_graph(epoch_data, corr_thresh)
    n_channels = epoch_data.shape[0]
    
    fig = plt.figure(figsize=(16, 6))
    
    # Plot 1: Graph structure
    ax1 = plt.subplot(1, 2, 1)
    pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
    
    # Node colors based on channel position
    node_colors = []
    for node in G.nodes():
        if node < 2:
            node_colors.append('red')
        elif node < 10:
            node_colors.append('green')
        elif node < 18:
            node_colors.append('blue')
        else:
            node_colors.append('orange')
    
    nx.draw_networkx_nodes(G, pos, node_size=600, node_color=node_colors, ax=ax1, alpha=0.8)
    nx.draw_networkx_labels(G, pos, labels={i: channel_names[i] for i in G.nodes()}, 
                           font_size=8, ax=ax1)
    
    edges = G.edges()
    if len(edges) > 0:
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=2, edge_color=weights, 
                              edge_cmap=plt.cm.viridis, edge_vmin=corr_thresh, 
                              edge_vmax=1.0, ax=ax1)
    
    ax1.set_title(f'{label} Epoch: EEG Connectivity Graph (Corr Thresh={corr_thresh})', 
                 fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Plot 2: Correlation heatmap
    ax2 = plt.subplot(1, 2, 2)
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1, 
               xticklabels=channel_names, yticklabels=channel_names, 
               cbar_kws={'label': 'Correlation'}, ax=ax2, square=True)
    ax2.set_title(f'{label} Epoch: Correlation Matrix', fontsize=12, fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'graph_structure_{label.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    num_edges = len(G.edges())
    avg_degree = 2 * num_edges / n_channels if n_channels > 0 else 0
    print(f"\n{label} Epoch Statistics:")
    print(f"  Number of nodes: {G.number_of_nodes()}")
    print(f"  Number of edges: {num_edges}")
    print(f"  Average degree: {avg_degree:.2f}")
    print(f"  Network density: {nx.density(G):.4f}")
    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        print(f"  Largest connected component size: {len(largest_cc)}")

# Get example epochs
stay_idx = np.where(y == 0)[0][0]
leave_idx = np.where(y == 1)[0][0]

print(f"\nStay epoch index: {stay_idx}")
print(f"Leave epoch index: {leave_idx}")

# Plot graphs
print("\n--- Plotting with correlation threshold 0.7 ---")
plot_eeg_graph_and_corr_heatmap(X[stay_idx], 'Stay', eeg_channels, corr_thresh=0.7)
plot_eeg_graph_and_corr_heatmap(X[leave_idx], 'Leave', eeg_channels, corr_thresh=0.7)

# ================================================================
# 3. GRAPH CONSTRUCTION FOR GNN
# ================================================================
print("\n" + "="*60)
print("CONSTRUCTING GRAPHS FOR GNN TRAINING")
print("="*60)

def create_graph_with_correlation(epoch_arr, label, corr_thresh=0.7):
    """Create torch_geometric Data object from EEG epoch"""
    n_channels, n_freq = epoch_arr.shape
    node_feats = torch.tensor(epoch_arr, dtype=torch.float32)
    corr = np.corrcoef(epoch_arr)
    edge_index = []
    edge_weights = []
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            if abs(corr[i, j]) > corr_thresh:
                edge_index.extend([[i, j], [j, i]])
                edge_weights.extend([abs(corr[i, j]), abs(corr[i, j])])
    if not edge_index:
        edge_index = [[i, i] for i in range(n_channels)]
        edge_weights = [1.0] * n_channels
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
    return GeoData(x=node_feats, edge_index=edge_index, edge_attr=edge_weights, 
                  y=torch.tensor([label], dtype=torch.long))

# ================================================================
# 4. SPATIO-TEMPORAL GNN MODEL
# ================================================================
class SpatioTemporalGNN(nn.Module):
    """Graph Neural Network with temporal modeling via RNN"""
    def __init__(self, node_feat_dim, hidden_dim=64, num_classes=2, 
                 num_gnn_layers=2, num_rnn_layers=2, K=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_rnn_layers = num_rnn_layers
        
        self.conv1 = ChebConv(node_feat_dim, hidden_dim, K=K)
        self.conv2 = ChebConv(hidden_dim, hidden_dim, K=K)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_rnn_layers, 
                         batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, data, return_features=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x_graph = global_mean_pool(x2, batch)
        x_temporal = x_graph.unsqueeze(1)
        _, h_n = self.rnn(x_temporal)
        x_final = h_n[-1]
        out = self.classifier(x_final)
        
        if return_features:
            return out, {'spatial': x2, 'temporal': x_final}
        return out

# ================================================================
# 5. K-FOLD CROSS-VALIDATION TRAINING
# ================================================================
print("\n" + "="*60)
print("K-FOLD CROSS-VALIDATION WITH SPATIO-TEMPORAL GNN")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

metrics = {'accuracy': [], 'auc': [], 'f1': [], 'fold_losses': []}

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold+1}/{n_splits}")
    print("-" * 40)
    
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]
    
    X_train_2d = X_train_fold.reshape(len(X_train_fold), -1)
    ros = RandomOverSampler(random_state=42)
    X_train_bal, y_train_bal = ros.fit_resample(X_train_2d, y_train_fold)
    X_train_bal = X_train_bal.reshape(-1, n_channels, n_freq)
    
    train_graphs = [create_graph_with_correlation(arr, label) for arr, label in zip(X_train_bal, y_train_bal)]
    test_graphs = [create_graph_with_correlation(arr, label) for arr, label in zip(X_test_fold, y_test_fold)]
    
    # CORRECTED: Use torch_geometric DataLoader
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)
    
    model = SpatioTemporalGNN(node_feat_dim=n_freq, hidden_dim=64, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    epoch_losses = []
    for epoch in range(20):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_losses.append(running_loss / len(train_loader))
    
    metrics['fold_losses'].append(epoch_losses)
    
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            prob = torch.softmax(out, dim=1)[:, 1]
            pred = out.argmax(dim=1)
            all_probs.extend(prob.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.view(-1).cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    auc_score = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    
    metrics['accuracy'].append(acc)
    metrics['auc'].append(auc_score)
    metrics['f1'].append(f1)
    
    print(f"Acc: {acc:.4f}, AUC: {auc_score:.4f}, F1: {f1:.4f}")

# ================================================================
# 6. PLOT RESULTS
# ================================================================
print("\n" + "="*60)
print("PLOTTING RESULTS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(range(1, n_splits+1), metrics['accuracy'], 'o-', linewidth=2, markersize=8)
axes[0, 0].set_title('Accuracy across Folds', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Fold')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0, 1])

axes[0, 1].plot(range(1, n_splits+1), metrics['auc'], 'o-', color='orange', linewidth=2, markersize=8)
axes[0, 1].set_title('AUC across Folds', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Fold')
axes[0, 1].set_ylabel('AUC')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim([0, 1])

axes[1, 0].plot(range(1, n_splits+1), metrics['f1'], 'o-', color='green', linewidth=2, markersize=8)
axes[1, 0].set_title('F1-Score across Folds', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Fold')
axes[1, 0].set_ylabel('F1-Score')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([0, 1])

for fold_idx, fold_losses in enumerate(metrics['fold_losses']):
    axes[1, 1].plot(range(1, len(fold_losses)+1), fold_losses, label=f'Fold {fold_idx+1}', alpha=0.7)
axes[1, 1].set_title('Training Loss across Epochs', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spatiotemporal_gnn_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Mean Accuracy: {np.mean(metrics['accuracy']):.4f} ± {np.std(metrics['accuracy']):.4f}")
print(f"Mean AUC: {np.mean(metrics['auc']):.4f} ± {np.std(metrics['auc']):.4f}")
print(f"Mean F1: {np.mean(metrics['f1']):.4f} ± {np.std(metrics['f1']):.4f}")

print("\nAll visualizations and results saved!")
