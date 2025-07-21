import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report
import matplotlib.pyplot as plt

# 1. Sliding Window Epoch Extraction for All Files
def load_all_epochs(folder, label, epoch_size=1000, overlap=0.5):
    all_epochs = []
    for fname in os.listdir(folder):
        if fname.endswith('.csv'):
            file_path = os.path.join(folder, fname)
            df = pd.read_csv(file_path)
            data = df.select_dtypes(include=[np.number]).values.T  # (channels, time)
            step = int(epoch_size * (1 - overlap))
            for start in range(0, data.shape[1] - epoch_size + 1, step):
                epoch = data[:, start:start+epoch_size]
                all_epochs.append((epoch, label))
    return all_epochs

# 2. Load all epochs from both classes
low_dir = 'Anx/env1/Low'
high_dir = 'Anx/env1/High'
epoch_size = 1000
overlap = 0.5

data_low = load_all_epochs(low_dir, label=0, epoch_size=epoch_size, overlap=overlap)
data_high = load_all_epochs(high_dir, label=1, epoch_size=epoch_size, overlap=overlap)
all_data = data_low + data_high
print(f"Loaded {len(all_data)} epochs (sliding window, all files).")

# 3. Convert each epoch to a graph
def epoch_to_graph(epoch_arr, label, corr_threshold=0.7):
    n_channels, _ = epoch_arr.shape
    node_feats = torch.tensor(epoch_arr, dtype=torch.float32)
    corr = np.corrcoef(epoch_arr)
    edge_index = []
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            if abs(corr[i, j]) > corr_threshold:
                edge_index.extend([[i, j], [j, i]])
    if not edge_index:
        edge_index = [[i, i] for i in range(n_channels)]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=node_feats, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))

graphs = [epoch_to_graph(arr, y) for arr, y in all_data]

# 4. Epoch-level Train/Test Split
labels = [g.y.item() for g in graphs]
train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42, stratify=labels)
train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=16)

# 5. GAT Model
class EEGGATNet(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim=32, num_classes=2, heads=4):
        super().__init__()
        self.gat1 = GATConv(node_feat_dim, hidden_dim, heads=heads)
        self.gat2 = GATConv(hidden_dim*heads, hidden_dim, heads=1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.classifier(x)

# 6. Training and Evaluation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EEGGATNet(node_feat_dim=epoch_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_probs.extend(probs[:,1].cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch.y.view(-1).cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {acc:.4f}")
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "High"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
print(classification_report(all_labels, all_preds, target_names=['Low', 'High']))
