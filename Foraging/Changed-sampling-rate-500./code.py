import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from imblearn.over_sampling import RandomOverSampler
from torch_geometric.data import Data as GeoData, DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, f1_score, accuracy_score
import matplotlib.pyplot as plt

# ========================
# 1. Data Loading with CORRECTED SAMPLING RATE
# ========================
input_folder = 'separated_classes'
eeg_channels = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'IO',
    'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6'
]

# CORRECTED: Actual sampling rate is 500 Hz
sampling_rate = 500  

max_time = 0
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

# Calculate frequency bins for 0-50 Hz range
freq_resolution = sampling_rate / max_time
print(f"Frequency resolution: {freq_resolution:.4f} Hz")
max_freq = 50
freq_bins = int(max_freq / freq_resolution) + 1
print(f"Using {freq_bins} frequency bins to cover 0-{max_freq} Hz")

all_X, all_y = [], []
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
print(f"Data shape: {X.shape}")

# ========================
# 2. Dataset class
# ========================
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ========================
# 3. CORRECTED EEGNet Architecture (based on original paper)
# ========================
class EEGNet(nn.Module):
    def __init__(self, chans=28, samples=100, num_classes=2, F1=8, F2=16, D=2, dropout=0.5):
        super().__init__()
        # Block 1: Temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        
        # Depthwise convolution (spatial filtering)
        self.depthwise = nn.Conv2d(F1, F1 * D, (chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.activation1 = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        
        # Block 2: Separable convolution
        self.separable = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)
        
        # Calculate flatten size
        with torch.no_grad():
            x_dummy = torch.zeros(1, 1, chans, samples)
            x_dummy = self.batchnorm1(self.conv1(x_dummy))
            x_dummy = self.activation1(self.batchnorm2(self.depthwise(x_dummy)))
            x_dummy = self.dropout1(self.avgpool1(x_dummy))
            x_dummy = self.activation2(self.batchnorm3(self.separable(x_dummy)))
            x_dummy = self.dropout2(self.avgpool2(x_dummy))
            self.flatten_size = x_dummy.numel()
        
        # Classification layer
        self.fc = nn.Linear(self.flatten_size, num_classes)
    
    def forward(self, x, return_features=False):
        # Block 1
        x1 = self.batchnorm1(self.conv1(x))  # Temporal features
        x2 = self.depthwise(x1)
        x2 = self.activation1(self.batchnorm2(x2))  # Spatial features
        x2 = self.dropout1(self.avgpool1(x2))
        
        # Block 2
        x3 = self.separable(x2)
        x3 = self.activation2(self.batchnorm3(x3))
        x3 = self.dropout2(self.avgpool2(x3))
        
        # Flatten
        x_flat = x3.view(x3.size(0), -1)
        out = self.fc(x_flat)
        
        if return_features:
            return out, {'temporal': x1, 'spatial': x2, 'final': x3}
        return out

# ========================
# 4. GCN Model
# ========================
class GCN(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim=32, num_classes=2):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, data, return_features=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = F.relu(self.conv1(x, edge_index))  # Layer 1 features
        x2 = F.relu(self.conv2(x1, edge_index))  # Layer 2 features
        x_pooled = global_mean_pool(x2, batch)
        out = self.classifier(x_pooled)
        
        if return_features:
            return out, {'layer1': x1, 'layer2': x2}
        return out

# ========================
# 5. Helper function to create graphs
# ========================
def create_graph(epoch_arr, label):
    n_chans, _ = epoch_arr.shape
    node_feats = torch.tensor(epoch_arr, dtype=torch.float32)
    corr = np.corrcoef(epoch_arr)
    edge_index = []
    for i in range(n_chans):
        for j in range(i+1, n_chans):
            if abs(corr[i,j]) > 0.7:
                edge_index.extend([[i,j], [j,i]])
    if not edge_index:
        edge_index = [[i,i] for i in range(n_chans)]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return GeoData(x=node_feats, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))

# ========================
# 6. K-Fold Cross-Validation with Metrics Tracking
# ========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Storage for metrics
eegnet_metrics = {'accuracy': [], 'auc': [], 'f1': [], 'fold_losses': []}
gcn_metrics = {'accuracy': [], 'auc': [], 'f1': [], 'fold_losses': []}

n_channels = X.shape[1]
n_freq = X.shape[2]

print("\n" + "="*60)
print("Starting K-Fold Cross-Validation")
print("="*60)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold+1}/{n_splits}")
    print("-" * 40)
    
    # Split data
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]
    
    # Oversample training set
    X_train_2d = X_train_fold.reshape(len(X_train_fold), -1)
    ros = RandomOverSampler(random_state=42)
    X_train_bal, y_train_bal = ros.fit_resample(X_train_2d, y_train_fold)
    X_train_bal = X_train_bal.reshape(-1, n_channels, n_freq)
    
    # ===== Train EEGNet =====
    print("Training EEGNet...")
    X_train_eegnet = X_train_bal[:, np.newaxis, :, :]
    X_test_eegnet = X_test_fold[:, np.newaxis, :, :]
    
    train_dataset = EEGDataset(X_train_eegnet, y_train_bal)
    test_dataset = EEGDataset(X_test_eegnet, y_test_fold)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    eegnet = EEGNet(chans=n_channels, samples=n_freq).to(device)
    optimizer = torch.optim.Adam(eegnet.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    epoch_losses = []
    for epoch in range(20):
        eegnet.train()
        running_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = eegnet(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_losses.append(running_loss / len(train_loader))
    
    eegnet_metrics['fold_losses'].append(epoch_losses)
    
    # Evaluate EEGNet
    eegnet.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = eegnet(Xb)
            prob = torch.softmax(out, dim=1)[:, 1]
            pred = out.argmax(dim=1)
            all_probs.extend(prob.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = auc(fpr, tpr)
    f1 = f1_score(all_labels, all_preds)
    
    eegnet_metrics['accuracy'].append(acc)
    eegnet_metrics['auc'].append(auc_score)
    eegnet_metrics['f1'].append(f1)
    
    print(f"EEGNet - Acc: {acc:.4f}, AUC: {auc_score:.4f}, F1: {f1:.4f}")
    
    # ===== Train GCN =====
    print("Training GCN...")
    train_graphs = [create_graph(arr, label) for arr, label in zip(X_train_bal, y_train_bal)]
    test_graphs = [create_graph(arr, label) for arr, label in zip(X_test_fold, y_test_fold)]
    
    train_loader_gcn = GeoDataLoader(train_graphs, batch_size=16, shuffle=True)
    test_loader_gcn = GeoDataLoader(test_graphs, batch_size=16, shuffle=False)
    
    gcn = GCN(node_feat_dim=n_freq, num_classes=2).to(device)
    optimizer = torch.optim.Adam(gcn.parameters(), lr=1e-4)
    
    epoch_losses = []
    for epoch in range(20):
        gcn.train()
        running_loss = 0.0
        for batch in train_loader_gcn:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = gcn(batch)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_losses.append(running_loss / len(train_loader_gcn))
    
    gcn_metrics['fold_losses'].append(epoch_losses)
    
    # Evaluate GCN
    gcn.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in test_loader_gcn:
            batch = batch.to(device)
            out = gcn(batch)
            prob = torch.softmax(out, dim=1)[:, 1]
            pred = out.argmax(dim=1)
            all_probs.extend(prob.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.view(-1).cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = auc(fpr, tpr)
    f1 = f1_score(all_labels, all_preds)
    
    gcn_metrics['accuracy'].append(acc)
    gcn_metrics['auc'].append(auc_score)
    gcn_metrics['f1'].append(f1)
    
    print(f"GCN - Acc: {acc:.4f}, AUC: {auc_score:.4f}, F1: {f1:.4f}")

# ========================
# 7. Plot K-Fold Results
# ========================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# EEGNet plots
axes[0, 0].plot(range(1, n_splits+1), eegnet_metrics['accuracy'], 'o-', label='EEGNet')
axes[0, 0].set_title('EEGNet: Accuracy across Folds')
axes[0, 0].set_xlabel('Fold')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].grid(True)

axes[0, 1].plot(range(1, n_splits+1), eegnet_metrics['auc'], 'o-', label='EEGNet', color='orange')
axes[0, 1].set_title('EEGNet: AUC across Folds')
axes[0, 1].set_xlabel('Fold')
axes[0, 1].set_ylabel('AUC')
axes[0, 1].grid(True)

axes[0, 2].plot(range(1, n_splits+1), eegnet_metrics['f1'], 'o-', label='EEGNet', color='green')
axes[0, 2].set_title('EEGNet: F1-Score across Folds')
axes[0, 2].set_xlabel('Fold')
axes[0, 2].set_ylabel('F1-Score')
axes[0, 2].grid(True)

# GCN plots
axes[1, 0].plot(range(1, n_splits+1), gcn_metrics['accuracy'], 'o-', label='GCN')
axes[1, 0].set_title('GCN: Accuracy across Folds')
axes[1, 0].set_xlabel('Fold')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].grid(True)

axes[1, 1].plot(range(1, n_splits+1), gcn_metrics['auc'], 'o-', label='GCN', color='orange')
axes[1, 1].set_title('GCN: AUC across Folds')
axes[1, 1].set_xlabel('Fold')
axes[1, 1].set_ylabel('AUC')
axes[1, 1].grid(True)

axes[1, 2].plot(range(1, n_splits+1), gcn_metrics['f1'], 'o-', label='GCN', color='green')
axes[1, 2].set_title('GCN: F1-Score across Folds')
axes[1, 2].set_xlabel('Fold')
axes[1, 2].set_ylabel('F1-Score')
axes[1, 2].grid(True)

plt.tight_layout()
plt.savefig('kfold_metrics.png', dpi=300)
plt.show()

# Plot training loss curves (epochs vs loss)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for fold_idx, fold_losses in enumerate(eegnet_metrics['fold_losses']):
    axes[0].plot(range(1, len(fold_losses)+1), fold_losses, label=f'Fold {fold_idx+1}')
axes[0].set_title('EEGNet: Training Loss across Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

for fold_idx, fold_losses in enumerate(gcn_metrics['fold_losses']):
    axes[1].plot(range(1, len(fold_losses)+1), fold_losses, label=f'Fold {fold_idx+1}')
axes[1].set_title('GCN: Training Loss across Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_loss.png', dpi=300)
plt.show()

print("\n" + "="*60)
print("Summary Statistics")
print("="*60)
print(f"EEGNet - Mean Accuracy: {np.mean(eegnet_metrics['accuracy']):.4f} ± {np.std(eegnet_metrics['accuracy']):.4f}")
print(f"EEGNet - Mean AUC: {np.mean(eegnet_metrics['auc']):.4f} ± {np.std(eegnet_metrics['auc']):.4f}")
print(f"EEGNet - Mean F1: {np.mean(eegnet_metrics['f1']):.4f} ± {np.std(eegnet_metrics['f1']):.4f}")
print(f"\nGCN - Mean Accuracy: {np.mean(gcn_metrics['accuracy']):.4f} ± {np.std(gcn_metrics['accuracy']):.4f}")
print(f"GCN - Mean AUC: {np.mean(gcn_metrics['auc']):.4f} ± {np.std(gcn_metrics['auc']):.4f}")
print(f"GCN - Mean F1: {np.mean(gcn_metrics['f1']):.4f} ± {np.std(gcn_metrics['f1']):.4f}")

# ========================
# 8. Visualize Layer-wise Features (for last fold's models)
# ========================
print("\n" + "="*60)
print("Visualizing Layer-wise Features")
print("="*60)

# Get a sample from test set
sample_idx = 0
sample_eegnet = torch.tensor(X_test_eegnet[sample_idx:sample_idx+1], dtype=torch.float32).to(device)

# EEGNet features
eegnet.eval()
with torch.no_grad():
    _, features_eegnet = eegnet(sample_eegnet, return_features=True)

# Plot EEGNet features
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Temporal features (layer 1)
temporal = features_eegnet['temporal'][0].cpu().numpy()
axes[0].imshow(temporal.mean(axis=0), aspect='auto', cmap='viridis')
axes[0].set_title('EEGNet: Temporal/Frequency Features (Layer 1)')
axes[0].set_xlabel('Time/Frequency')
axes[0].set_ylabel('Channels')

# Spatial features (layer 2)
spatial = features_eegnet['spatial'][0].cpu().numpy()
axes[1].imshow(spatial.mean(axis=0), aspect='auto', cmap='viridis')
axes[1].set_title('EEGNet: Spatial Features (Layer 2)')
axes[1].set_xlabel('Time/Frequency')
axes[1].set_ylabel('Features')

# Final features
final = features_eegnet['final'][0].cpu().numpy()
axes[2].imshow(final.mean(axis=0), aspect='auto', cmap='viridis')
axes[2].set_title('EEGNet: Final Features')
axes[2].set_xlabel('Time/Frequency')
axes[2].set_ylabel('Features')

plt.tight_layout()
plt.savefig('eegnet_features.png', dpi=300)
plt.show()

# GCN features
sample_graph = test_graphs[sample_idx]
sample_graph = sample_graph.to(device)

gcn.eval()
with torch.no_grad():
    _, features_gcn = gcn(sample_graph, return_features=True)

# Plot GCN features
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

layer1 = features_gcn['layer1'].cpu().numpy()
axes[0].imshow(layer1, aspect='auto', cmap='viridis')
axes[0].set_title('GCN: Layer 1 Features')
axes[0].set_xlabel('Feature Dimension')
axes[0].set_ylabel('Nodes (Channels)')

layer2 = features_gcn['layer2'].cpu().numpy()
axes[1].imshow(layer2, aspect='auto', cmap='viridis')
axes[1].set_title('GCN: Layer 2 Features')
axes[1].set_xlabel('Feature Dimension')
axes[1].set_ylabel('Nodes (Channels)')

plt.tight_layout()
plt.savefig('gcn_features.png', dpi=300)
plt.show()

print("\nAll visualizations saved!")
