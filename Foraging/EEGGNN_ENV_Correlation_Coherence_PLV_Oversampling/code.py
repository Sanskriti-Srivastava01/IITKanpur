import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import Sequential, GINConv, global_mean_pool
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
import gc
import scipy.signal

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Channel configuration (update to match your data)
EXPECTED_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                     'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'IO',
                     'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6']

def load_env_data(folder, env, downsample=10, target_time=1000):
    """Load EEG data for a given environment."""
    X_list, y_list = [], []
    files = [f for f in os.listdir(folder) if f.endswith('.csv') and f'env{env}' in f]
    if not files:
        print(f"No files found for env{env}")
        return np.zeros((0, len(EXPECTED_CHANNELS), target_time // downsample)), np.zeros(0)
    target_samples = target_time // downsample
    for fname in files:
        try:
            df = pd.read_csv(os.path.join(folder, fname), header=0)
            if df.empty:
                continue
            df = df.reindex(columns=EXPECTED_CHANNELS).fillna(0)
            arr = df.values.T.astype(np.float32)
            if downsample > 1:
                arr = arr[:, ::downsample]
            if arr.shape[1] > target_samples:
                arr = arr[:, :target_samples]
            elif arr.shape[1] < target_samples:
                arr = np.pad(arr, ((0,0), (0,target_samples-arr.shape[1])), mode='constant')
            X_list.append(arr)
            y_list.append(0 if 'stay' in fname else 1)
        except Exception as e:
            print(f"Skipped {fname}: {str(e)}")
            continue
    if not X_list:
        return np.zeros((0, len(EXPECTED_CHANNELS), target_samples)), np.zeros(0)
    X = np.stack(X_list)
    y = np.array(y_list)
    print(f"ENV{env}: Loaded {X.shape[0]} trials (Stay: {sum(y==0)}, Leave: {sum(y==1)})")
    return X, y

def oversample_minority(graphs, labels):
    """Random oversample minority class graphs."""
    label_counts = Counter(labels)
    if len(label_counts) < 2:
        return graphs, labels
    majority_label = max(label_counts, key=label_counts.get)
    minority_label = 1 - majority_label
    minority_indices = [i for i, label in enumerate(labels) if label == minority_label]
    majority_indices = [i for i, label in enumerate(labels) if label == majority_label]
    num_to_add = len(majority_indices) - len(minority_indices)
    if num_to_add <= 0:
        return graphs, labels
    sampled_indices = np.random.choice(minority_indices, size=num_to_add, replace=True)
    oversampled_graphs = graphs + [graphs[i].clone() for i in sampled_indices]
    oversampled_labels = labels + [labels[i] for i in sampled_indices]
    print("After oversampling, class counts:", Counter(oversampled_labels))
    return oversampled_graphs, oversampled_labels

def compute_plv(X_trial):
    """Phase Locking Value (PLV) between all channel pairs."""
    n_channels = X_trial.shape[0]
    plv_matrix = np.zeros((n_channels, n_channels))
    analytic_signal = scipy.signal.hilbert(X_trial, axis=1)
    phase = np.angle(analytic_signal)
    for i in range(n_channels):
        for j in range(n_channels):
            phase_diff = phase[i] - phase[j]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv_matrix[i,j] = plv
    return plv_matrix

def compute_coh(X_trial, fs=200, nperseg=256):
    """Coherence (COH) between all channel pairs."""
    n_channels = X_trial.shape[0]
    coh_matrix = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(n_channels):
            f, Cxy = scipy.signal.coherence(X_trial[i], X_trial[j], fs=fs, nperseg=nperseg)
            coh_matrix[i,j] = np.mean(Cxy)
    return coh_matrix

def save_connectivity_matrices(X_data, folder_path):
    """Save correlation, coherence, and PLV matrices for a dataset."""
    n_trials = X_data.shape[0]
    corr_matrices = []
    coh_matrices = []
    plv_matrices = []
    for i in range(n_trials):
        trial = X_data[i]
        corr = np.corrcoef(trial)
        coh = compute_coh(trial)
        plv = compute_plv(trial)
        corr_matrices.append(corr)
        coh_matrices.append(coh)
        plv_matrices.append(plv)
    corr_matrices = np.array(corr_matrices)
    coh_matrices = np.array(coh_matrices)
    plv_matrices = np.array(plv_matrices)
    os.makedirs(folder_path, exist_ok=True)
    np.save(os.path.join(folder_path, 'correlation_matrices.npy'), corr_matrices)
    np.save(os.path.join(folder_path, 'coherence_matrices.npy'), coh_matrices)
    np.save(os.path.join(folder_path, 'plv_matrices.npy'), plv_matrices)
    print(f"Saved connectivity matrices to {folder_path}")
    print("Shapes:", corr_matrices.shape, coh_matrices.shape, plv_matrices.shape)
    return corr_matrices.shape, coh_matrices.shape, plv_matrices.shape

def create_graphs(X, y, connectivity_type='corr', threshold=0.6):
    """Create graphs with correlation/coherence/PLV edges."""
    n_samples, n_channels, _ = X.shape
    graphs = []
    for i in range(n_samples):
        x = torch.tensor(X[i], dtype=torch.float32)
        if connectivity_type == 'plv':
            adj_matrix = compute_plv(X[i])
        elif connectivity_type == 'coh':
            adj_matrix = compute_coh(X[i])
        else:  # Default to correlation
            adj_matrix = np.corrcoef(X[i])
        adj_matrix = (abs(adj_matrix) > threshold).astype(int)
        edge_index = []
        for j in range(n_channels):
            for k in range(j+1, n_channels):
                if adj_matrix[j,k] == 1:
                    edge_index.extend([[j,k], [k,j]])
        if not edge_index:  # Fallback to self-loops
            edge_index = [[j,j] for j in range(n_channels)]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        graphs.append(Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([y[i]], dtype=torch.long)
        ))
    return graphs

class EEGGNNWithFeatures(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim=128):
        super().__init__()
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
        )
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index.to(data.x.device), data.batch.to(data.x.device)
        out1 = self.conv1(x, edge_index)
        out1_activated = F.relu(out1)
        out2 = self.conv2(out1_activated, edge_index)
        out2_activated = F.relu(out2)
        pooled = global_mean_pool(out2_activated, batch)
        logits = self.classifier(pooled)
        return logits, {'conv1': out1_activated, 'conv2': out2_activated, 'pooled': pooled}

def plot_layer_features(batch, model, device):
    """Plot feature maps for each layer of the model."""
    batch = batch.to(device)
    logits, features = model(batch)
    for layer_name, feat in features.items():
        feat = feat.cpu().detach().numpy()
        plt.figure(figsize=(12, 4))
        if layer_name == 'pooled':
            # For pooled features: plot mean across features for each sample
            if feat.ndim == 2:
                feat = feat.mean(axis=1)  # (batch_size, hidden_dim) -> (batch_size,)
            plt.title(f"{layer_name} (shape: {feat.shape})")
            plt.bar(range(len(feat)), feat)
        else:
            # For conv1/conv2 features: plot average across batch if needed
            if feat.ndim > 2:
                feat = feat.mean(axis=0)
            plt.title(f"{layer_name} (shape: {feat.shape})")
            plt.imshow(feat, aspect='auto', cmap='viridis')
            plt.colorbar()
        plt.show()
    return logits, features

def train_and_evaluate_env(X, y, env_name, batch_size=2, epochs=200, connectivity_type='corr', threshold=0.6):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_accs = []
    all_test_labels, all_test_preds, all_test_probs = [], [], []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    print(f"\n{'='*50}")
    print(f"Training GNN for {env_name} with {connectivity_type} connectivity")
    print(f"{'='*50}")
    for split, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n=== {env_name} Split {split+1}/5 ===")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        train_graphs = create_graphs(X_train, y_train, connectivity_type, threshold)
        test_graphs = create_graphs(X_test, y_test, connectivity_type, threshold)
        train_labels = [g.y.item() for g in train_graphs]
        train_graphs, train_labels = oversample_minority(train_graphs, train_labels)
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=batch_size)
        model = EEGGNNWithFeatures(node_feat_dim=X_train.shape[2]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                with autocast():
                    logits, _ = model(batch)
                    loss = criterion(logits, batch.y.view(-1).long())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
        model.eval()
        split_preds, split_probs, split_labels = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits, features = model(batch)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                split_probs.extend(probs[:,1].cpu().numpy())
                split_preds.extend(preds.cpu().numpy())
                split_labels.extend(batch.y.view(-1).long().cpu().numpy())
                if len(split_labels) == batch_size:
                    plot_layer_features(batch, model, device)
        all_test_labels.extend(split_labels)
        all_test_preds.extend(split_preds)
        all_test_probs.extend(split_probs)
        acc = accuracy_score(split_labels, split_preds)
        test_accs.append(acc)
        print(f"Split {split+1} Accuracy: {acc:.4f}")
        cm = confusion_matrix(split_labels, split_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Leave"])
        disp.plot(cmap='Blues')
        plt.title(f"{env_name} {connectivity_type} Confusion Matrix - Split {split+1}")
        plt.show()
        fpr, tpr, _ = roc_curve(split_labels, split_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{env_name} {connectivity_type} ROC Curve - Split {split+1}')
        plt.legend(loc="lower right")
        plt.show()
        del X_train, y_train, X_test, y_test, train_graphs, test_graphs
        gc.collect()
        torch.cuda.empty_cache()
    print(f"\n{env_name} {connectivity_type} Average Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    cm = confusion_matrix(all_test_labels, all_test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Leave"])
    disp.plot(cmap='Blues')
    plt.title(f"{env_name} {connectivity_type} Overall Confusion Matrix")
    plt.show()
    fpr, tpr, _ = roc_curve(all_test_labels, all_test_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{env_name} {connectivity_type} Overall ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    return all_test_labels, all_test_preds, all_test_probs

def main(folder_path):
    print("Loading ENV1 data...")
    X_env1, y_env1 = load_env_data(folder_path, env='1', downsample=2)
    if len(X_env1) > 0:
        save_folder = 'connectivity_matrices_env1'
        print("\nSaving ENV1 connectivity matrices...")
        save_connectivity_matrices(X_env1, save_folder)
        for conn_type in ['corr', 'coh', 'plv']:
            print(f"\nRunning GNN with {conn_type} connectivity for ENV1...")
            env1_labels, env1_preds, env1_probs = train_and_evaluate_env(
                X_env1, y_env1, env_name="ENV1", batch_size=2, epochs=200,
                connectivity_type=conn_type, threshold=0.5
            )
    else:
        print("No data found for ENV1")
    print("\nLoading ENV2 data...")
    X_env2, y_env2 = load_env_data(folder_path, env='2', downsample=2)
    if len(X_env2) > 0:
        save_folder = 'connectivity_matrices_env2'
        print("\nSaving ENV2 connectivity matrices...")
        save_connectivity_matrices(X_env2, save_folder)
        for conn_type in ['corr', 'coh', 'plv']:
            print(f"\nRunning GNN with {conn_type} connectivity for ENV2...")
            env2_labels, env2_preds, env2_probs = train_and_evaluate_env(
                X_env2, y_env2, env_name="ENV2", batch_size=2, epochs=200,
                connectivity_type=conn_type, threshold=0.5
            )
    else:
        print("No data found for ENV2")
        
if __name__ == '__main__':
    folder_path = "separated_classes"  # Replace with your actual folder path
    main(folder_path)
