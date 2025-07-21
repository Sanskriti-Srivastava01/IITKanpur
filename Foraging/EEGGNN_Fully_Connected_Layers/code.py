def create_graphs_fully_connected(X, y):
    n_samples, n_channels, n_time = X.shape
    graphs = []
    for i in range(n_samples):
        x = torch.tensor(X[i], dtype=torch.float32)
        # Create fully connected edges (no self-loops)
        edge_index = []
        for j in range(n_channels):
            for k in range(n_channels):
                if j != k:
                    edge_index.append([j, k])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        graphs.append(Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([y[i]], dtype=torch.long)
        ))
    return graphs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

class EEGGNN(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim=16):
        super().__init__()
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        )
        self.classifier = nn.Linear(hidden_dim, 2)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = global_mean_pool(x, data.batch)
        return self.classifier(x)
import numpy as np
from torch_geometric.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import os
import gc
from torch.cuda.amp import autocast, GradScaler

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def load_split_data(folder, split_indices, downsample=10):
    X_parts, y_parts = [], []
    for i in split_indices:
        X = np.load(os.path.join(folder, f"X_split_bal_{i}.npy")).astype(np.float32)
        X = X[:, :, ::downsample]
        y = np.load(os.path.join(folder, f"y_split_bal_{i}.npy"))
        X_parts.append(X)
        y_parts.append(y)
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)

def train_and_evaluate_split(folder, batch_size=1, epochs=20, downsample=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_accs = []
    accum_steps = 4

    for split in range(10):
        print(f"\n=== Processing Split {split+1}/10 ===")
        train_indices = [i for i in range(10) if i != split]
        X_train, y_train = load_split_data(folder, train_indices, downsample)
        X_test = np.load(os.path.join(folder, f"X_split_bal_{split}.npy")).astype(np.float32)[:, :, ::downsample]
        y_test = np.load(os.path.join(folder, f"y_split_bal_{split}.npy"))
        assert len(X_train) == len(y_train), "X/y length mismatch in training data"
        assert len(X_test) == len(y_test), "X/y length mismatch in test data"
        assert not np.isnan(y_train).any(), "NaN values in training labels"
        assert not np.isnan(y_test).any(), "NaN values in test labels"
        train_graphs = create_graphs_fully_connected(X_train, y_train)
        test_graphs = create_graphs_fully_connected(X_test, y_test)
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=batch_size)
        model = EEGGNN(node_feat_dim=X_train.shape[2]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            optimizer.zero_grad()
            for i, batch in enumerate(train_loader):
                batch = batch.to(device)
                if batch.y.numel() == 0 or batch.edge_index.shape[1] == 0:
                    continue
                with autocast():
                    out = model(batch)
                    loss = criterion(out, batch.y.view(-1).long()) / accum_steps
                scaler.scale(loss).backward()
                if (i+1) % accum_steps == 0 or (i+1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                total_loss += loss.item() * accum_steps
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
        model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                if batch.y.numel() == 0 or batch.edge_index.shape[1] == 0:
                    continue
                with autocast():
                    logits = model(batch)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_probs.extend(probs[:,1].cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.view(-1).long().cpu().numpy())
        if len(all_labels) == 0:
            print("⚠️ No valid predictions for this split, skipping")
            continue
        acc = accuracy_score(all_labels, all_preds)
        test_accs.append(acc)
        print(f"Accuracy: {acc:.4f}")
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class0", "Class1"])
        disp.plot(cmap='Blues')
        plt.title(f"Confusion Matrix - Split {split+1}")
        plt.show()
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Split {split+1}')
        plt.legend(loc="lower right")
        plt.show()
        del X_train, y_train, X_test, y_test, train_graphs, test_graphs
        gc.collect()
        torch.cuda.empty_cache()
    if test_accs:
        print(f"\nAverage Test Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    else:
        print("\n⚠️ No valid splits completed.")

if __name__ == '__main__':
    train_and_evaluate_split(folder="balanced_splits", downsample=10)
