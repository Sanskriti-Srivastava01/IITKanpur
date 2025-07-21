import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class EEGGCN(nn.Module):
    def __init__(self, n_node_features, hidden_dim=32, n_classes=2):
        super().__init__()
        self.conv1 = GCNConv(n_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)  # Pool over nodes per graph
        return self.lin(x)
import numpy as np
from torch_geometric.data import Data, DataLoader

def build_fully_connected_edge_index(n_nodes):
    # All pairs except self-loops
    edge_index = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                edge_index.append([i, j])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

def eeg_to_graphs(X, y):
    # X: (n_samples, n_channels, n_time), y: (n_samples,)
    n_samples, n_channels, n_time = X.shape
    edge_index = build_fully_connected_edge_index(n_channels)
    data_list = []
    for i in range(n_samples):
        node_features = torch.tensor(X[i], dtype=torch.float32)  # shape: (n_channels, n_time)
        label = int(y[i])
        data = Data(x=node_features, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
        data_list.append(data)
    return data_list
import os
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_splits = 5
batch_size = 16
epochs = 20
test_accs = []

for test_split in range(num_splits):
    # Load splits
    X_test = np.load(f'equal_splits/X_split_{test_split}.npy')
    y_test = np.load(f'equal_splits/y_split_{test_split}.npy')
    X_train = np.concatenate([np.load(f'equal_splits/X_split_{i}.npy') for i in range(num_splits) if i != test_split])
    y_train = np.concatenate([np.load(f'equal_splits/y_split_{i}.npy') for i in range(num_splits) if i != test_split])

    # Build graphs
    train_graphs = eeg_to_graphs(X_train, y_train)
    test_graphs = eeg_to_graphs(X_test, y_test)
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)

    # Model
    n_node_features = X_train.shape[2]  # n_time
    model = EEGGCN(n_node_features=n_node_features, n_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    test_accs.append(acc)
    print(f"Fold {test_split+1}/{num_splits} Test Accuracy: {acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay (0)", "Leave (1)"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Fold {test_split+1}")
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = roc_auc_score(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Fold {test_split+1}')
    plt.legend(loc="lower right")
    plt.show()

print(f"\nAverage Test Accuracy across all folds: {np.mean(test_accs):.4f}")
