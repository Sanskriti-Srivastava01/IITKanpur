import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
import gc

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Channel configuration (update to match your data)
EXPECTED_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'IO',
                    'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6']

def load_env_data(folder, env, downsample=10, target_time=1000):
    """
    Robust EEG data loading with shape standardization.
    - Forces all files to (n_channels, target_time)
    - Handles missing channels and varying time lengths
    """
    X_list, y_list = [], []
    files = [f for f in os.listdir(folder) if f.endswith('.csv') and f'env{env}' in f]
    
    if not files:
        print(f"No files found for env{env}")
        return np.zeros((0, len(EXPECTED_CHANNELS), target_time // downsample)), np.zeros(0)

    target_samples = target_time // downsample
    
    for fname in files:
        try:
            # Read CSV and validate
            df = pd.read_csv(os.path.join(folder, fname), header=0)
            if df.empty:
                continue
                
            # Select and order channels, fill missing with zeros
            df = df.reindex(columns=EXPECTED_CHANNELS).fillna(0)
            
            # Downsample and standardize time dimension
            arr = df.values.T.astype(np.float32)
            if downsample > 1:
                arr = arr[:, ::downsample]
            
            # Pad/truncate to target_samples
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
    """Random oversample minority class graphs"""
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
    return oversampled_graphs, oversampled_labels

def create_graphs(X, y, corr_threshold=0.6):
    """Create PyG graphs with correlation-based edges"""
    n_samples, n_channels, n_time = X.shape
    graphs = []
    for i in range(n_samples):
        x = torch.tensor(X[i], dtype=torch.float32)
        corr_matrix = np.corrcoef(X[i])
        edge_index = []
        for j in range(n_channels):
            for k in range(j+1, n_channels):
                if abs(corr_matrix[j,k]) > corr_threshold:
                    edge_index.extend([[j,k], [k,j]])
        if not edge_index:  # Fallback to self-loops
            edge_index = [[j,j] for j in range(n_channels)]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        graphs.append(Data(x=x, edge_index=edge_index, y=torch.tensor([y[i]], dtype=torch.long)))
    return graphs

class EEGGNN(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim=32):
        super().__init__()
        self.conv1 = GINConv(nn.Sequential(nn.Linear(node_feat_dim, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim, hidden_dim)))
        self.classifier = nn.Linear(hidden_dim, 2)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = global_mean_pool(x, data.batch)
        return self.classifier(x)

def train_and_evaluate_env(X, y, env_name, batch_size=2, epochs=20, corr_threshold=0.6):
    """Train and evaluate GNN for a specific environment"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_accs = []
    all_test_labels, all_test_preds, all_test_probs = [], [], []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    print(f"\n{'='*50}")
    print(f"Training GNN for {env_name}")
    print(f"{'='*50}")
    for split, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n=== {env_name} Split {split+1}/5 ===")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        train_graphs = create_graphs(X_train, y_train, corr_threshold)
        test_graphs = create_graphs(X_test, y_test, corr_threshold)
        train_labels = [g.y.item() for g in train_graphs]
        train_graphs, train_labels = oversample_minority(train_graphs, train_labels)
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=batch_size)
        model = EEGGNN(node_feat_dim=X_train.shape[2]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                with autocast():
                    out = model(batch)
                    loss = criterion(out, batch.y.view(-1).long())
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
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                split_probs.extend(probs[:,1].cpu().numpy())
                split_preds.extend(preds.cpu().numpy())
                split_labels.extend(batch.y.view(-1).long().cpu().numpy())
        all_test_labels.extend(split_labels)
        all_test_preds.extend(split_preds)
        all_test_probs.extend(split_probs)
        acc = accuracy_score(split_labels, split_preds)
        test_accs.append(acc)
        print(f"Split {split+1} Accuracy: {acc:.4f}")
        cm = confusion_matrix(split_labels, split_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Leave"])
        disp.plot(cmap='Blues')
        plt.title(f"{env_name} Confusion Matrix - Split {split+1}")
        plt.show()
        fpr, tpr, _ = roc_curve(split_labels, split_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{env_name} ROC Curve - Split {split+1}')
        plt.legend(loc="lower right")
        plt.show()
        del X_train, y_train, X_test, y_test, train_graphs, test_graphs
        gc.collect()
        torch.cuda.empty_cache()
    print(f"\n{env_name} Average Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    cm = confusion_matrix(all_test_labels, all_test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Leave"])
    disp.plot(cmap='Blues')
    plt.title(f"{env_name} Overall Confusion Matrix")
    plt.show()
    fpr, tpr, _ = roc_curve(all_test_labels, all_test_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{env_name} Overall ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    return all_test_labels, all_test_preds, all_test_probs

def main(folder_path):
    print("Loading ENV1 data...")
    X_env1, y_env1 = load_env_data(folder_path, env='1', downsample=10)
    if len(X_env1) > 0:
        env1_labels, env1_preds, env1_probs = train_and_evaluate_env(X_env1, y_env1, env_name="ENV1", batch_size=2, epochs=20, corr_threshold=0.6)
    else:
        print("No data found for ENV1")
    print("\nLoading ENV2 data...")
    X_env2, y_env2 = load_env_data(folder_path, env='2', downsample=10)
    if len(X_env2) > 0:
        env2_labels, env2_preds, env2_probs = train_and_evaluate_env(X_env2, y_env2, env_name="ENV2", batch_size=2, epochs=20, corr_threshold=0.6)
    else:
        print("No data found for ENV2")

if __name__ == '__main__':
    folder_path = "separated_classes"  # Replace with your actual folder path
    main(folder_path)
