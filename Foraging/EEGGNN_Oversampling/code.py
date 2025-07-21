import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import os
import gc
from torch.cuda.amp import autocast, GradScaler
from collections import Counter

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
def oversample_minority(graphs, labels):
    """Random oversample minority class graphs"""
    label_counts = Counter(labels)
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
def load_data(folder, downsample=10):
    """Load all splits from the folder"""
    X_parts, y_parts = [], []
    i = 0
    while True:
        x_path = os.path.join(folder, f"X_split_bal_{i}.npy")
        y_path = os.path.join(folder, f"y_split_bal_{i}.npy")
        if not os.path.exists(x_path) or not os.path.exists(y_path):
            break
        X = np.load(x_path).astype(np.float32)
        y = np.load(y_path)
        X = X[:, :, ::downsample]
        X_parts.append(X)
        y_parts.append(y)
        i += 1
    X = np.concatenate(X_parts, axis=0) if X_parts else np.zeros((0, 0, 0))
    y = np.concatenate(y_parts, axis=0) if y_parts else np.zeros(0)
    return X, y
def train_and_evaluate(folder, batch_size=2, epochs=20, corr_threshold=0.6, downsample=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_accs = []
    # Initialize lists to collect results across all splits
    all_test_labels = []
    all_test_preds = []
    all_test_probs = []
    
    # Verify folder exists
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist")
    
    # Load data
    X, y = load_data(folder, downsample)
    print(f"Loaded {len(X)} samples from {folder}")
    if len(X) == 0:
        raise ValueError("No data loaded. Check file names and paths.")
    
    # K-Fold cross-validation
    kf = KFold(n_splits=10, shuffle=True)
    for split, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n=== Processing Split {split+1}/10 ===")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Create graphs
        train_graphs = create_graphs(X_train, y_train, corr_threshold)
        test_graphs = create_graphs(X_test, y_test, corr_threshold)
        
        # Oversample minority class
        train_labels = [g.y.item() for g in train_graphs]
        train_graphs, train_labels = oversample_minority(train_graphs, train_labels)
        
        # Data loaders
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=batch_size)

        # Model setup
        model = EEGGNN(node_feat_dim=X_train.shape[2]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler()

        # Training loop
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

        # Evaluation
        model.eval()
        split_preds = []
        split_probs = []
        split_labels = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                split_probs.extend(probs[:,1].cpu().numpy())
                split_preds.extend(preds.cpu().numpy())
                split_labels.extend(batch.y.view(-1).long().cpu().numpy())
        # Append to global lists
        all_test_labels.extend(split_labels)
        all_test_preds.extend(split_preds)
        all_test_probs.extend(split_probs)
        
        acc = accuracy_score(split_labels, split_preds)
        test_accs.append(acc)
        print(f"Split {split+1} Accuracy: {acc:.4f}")

        # Cleanup
        del X_train, y_train, X_test, y_test, train_graphs, test_graphs
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nAverage Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    return all_test_labels, all_test_preds, all_test_probs
# Run training and evaluation
all_labels, all_preds, all_probs = train_and_evaluate(folder="balanced_splits")
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
