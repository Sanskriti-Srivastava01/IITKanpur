import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import RandomOverSampler
from torch_geometric.data import Data as GeoData, DataLoader as GeoDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt

# 1. Data Loading with proper frequency resolution
input_folder = 'separated_classes'
eeg_channels = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'IO',
    'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6'
]

# IMPORTANT: Set your actual sampling rate here
sampling_rate = 256  # Replace with your actual EEG sampling rate (e.g., 128, 256, 512 Hz)

# Find max time length
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
# Frequency resolution = sampling_rate / max_time
freq_resolution = sampling_rate / max_time
print(f"Frequency resolution: {freq_resolution:.4f} Hz")

# Calculate how many bins we need for 0-50 Hz
max_freq = 100  # Hz
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
    
    # FFT with proper frequency range
    fft_vals = np.abs(np.fft.rfft(data_padded.T, axis=1))
    freq_feat = fft_vals[:, :freq_bins]  # Extract 0-50 Hz
    
    if file.endswith('_stay.csv'):
        all_X.append(freq_feat)
        all_y.append(0)
    elif file.endswith('_leave.csv'):
        all_X.append(freq_feat)
        all_y.append(1)

X = np.stack(all_X, axis=0)
y = np.array(all_y)

print(f"Data shape: {X.shape}")  # (n_epochs, n_channels, freq_bins)
print(f"Frequency bins from 0 to {max_freq} Hz: {freq_bins}")

perm = np.random.permutation(len(X))
X, y = X[perm], y[perm]

# 2. Train-Test Split & Oversampling (train only)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
n_samples, n_channels, n_freq = X_train.shape
X_train_2d = X_train.reshape(n_samples, -1)
ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train_2d, y_train)
X_train_bal = X_train_bal.reshape(-1, n_channels, n_freq)

# 3. Dataset classes
class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# EEGNet needs extra channel dimension
X_train_eegnet = X_train_bal[:, np.newaxis, :, :]
X_test_eegnet = X_test[:, np.newaxis, :, :]
train_dataset_eegnet = EEGDataset(X_train_eegnet, y_train_bal)
test_dataset_eegnet = EEGDataset(X_test_eegnet, y_test)
train_loader_eegnet = DataLoader(train_dataset_eegnet, batch_size=16, shuffle=True)
test_loader_eegnet = DataLoader(test_dataset_eegnet, batch_size=16, shuffle=False)

# LMDA and GCN
train_dataset_lmda = EEGDataset(X_train_bal, y_train_bal)
test_dataset_lmda = EEGDataset(X_test, y_test)
train_loader_lmda = DataLoader(train_dataset_lmda, batch_size=16, shuffle=True)
test_loader_lmda = DataLoader(test_dataset_lmda, batch_size=16, shuffle=False)

# 4. Model definitions (same as before)
class EEGNet(nn.Module):
    def __init__(self, chans=28, freq_bins=100, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, min(64, freq_bins)), padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.depthwise = nn.Conv2d(16, 32, (chans, 1), groups=16)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (1, min(16, freq_bins)), padding='same')
        self.bn3 = nn.BatchNorm2d(32)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)
    def forward(self, x):
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.depthwise(x)))
        x = F.elu(self.bn3(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class LMDA(nn.Module):
    def __init__(self, chans=28, freq_bins=100, num_classes=2, depth=4, kernel=45, channel_depth1=8, channel_depth2=8, dropout_rate=0.5):
        super().__init__()
        self.depth = depth
        self.dropout = nn.Dropout(dropout_rate)
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, chans))
        nn.init.xavier_uniform_(self.channel_weight.data)
        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, (1, min(kernel, freq_bins)), padding=(0, min(kernel, freq_bins)//2), bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
            nn.AvgPool2d((1,4)),
            nn.Dropout(dropout_rate/2)
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, (chans,1), groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
            nn.AvgPool2d((1,8)),
            nn.Dropout(dropout_rate/2)
        )
        with torch.no_grad():
            dummy = torch.ones(1, depth, chans, freq_bins)
            out = torch.einsum('bdcw,hdc->bhcw', dummy, self.channel_weight)
            out = self.time_conv(out)
            out = self.channel_conv(out)
            self.fc_input = out.numel()
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    def EEGDepthAttention(self, x):
        N, C, H, W = x.size()
        k = 7
        adaptive_pool = nn.AdaptiveAvgPool2d((1,W))
        conv = nn.Conv2d(1, 1, kernel_size=(k,1), padding=(k//2,0), bias=True).to(x.device)
        softmax = nn.Softmax(dim=-2)
        x_pool = adaptive_pool(x)
        x_transpose = x_pool.transpose(-2,-3)
        y = conv(x_transpose)
        y = softmax(y)
        y = y.transpose(-2,-3)
        return y * C * x
    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.depth, 1, 1)
        x = torch.einsum('bdcw,hdc->bhcw', x, self.channel_weight)
        x = self.time_conv(x)
        x = self.EEGDepthAttention(x)
        x = self.dropout(x)
        x = self.channel_conv(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class GCN(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim=32, num_classes=2):
        super().__init__()
        self.conv1 = GCNConv(node_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.classifier(x)

# 5. Training and Evaluation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
epochs = 20

# EEGNet
print("\nTraining EEGNet...")
eegnet = EEGNet(chans=n_channels, freq_bins=n_freq).to(device)
optimizer = torch.optim.Adam(eegnet.parameters(), lr=1e-4)
for epoch in range(epochs):
    eegnet.train()
    for Xb, yb in train_loader_eegnet:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = eegnet(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

eegnet.eval()
all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for Xb, yb in test_loader_eegnet:
        Xb, yb = Xb.to(device), yb.to(device)
        out = eegnet(Xb)
        prob = torch.softmax(out, dim=1)[:, 1]
        pred = out.argmax(dim=1)
        all_probs.extend(prob.cpu().numpy())
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

print("EEGNet Results:")
cm = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(cm, display_labels=["Stay", "Leave"]).plot(cmap=plt.cm.Blues)
plt.title("EEGNet Confusion Matrix")
plt.show()
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1],'--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('EEGNet ROC')
plt.legend()
plt.show()
print(f"EEGNet ROC AUC: {roc_auc:.4f}")

# LMDA
print("\nTraining LMDA...")
lmda = LMDA(chans=n_channels, freq_bins=n_freq).to(device)
optimizer = torch.optim.Adam(lmda.parameters(), lr=1e-4)
for epoch in range(epochs):
    lmda.train()
    for Xb, yb in train_loader_lmda:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = lmda(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

lmda.eval()
all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for Xb, yb in test_loader_lmda:
        Xb, yb = Xb.to(device), yb.to(device)
        out = lmda(Xb)
        prob = torch.softmax(out, dim=1)[:,1]
        pred = out.argmax(dim=1)
        all_probs.extend(prob.cpu().numpy())
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

print("LMDA Results:")
cm = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(cm, display_labels=["Stay", "Leave"]).plot(cmap=plt.cm.Blues)
plt.title("LMDA Confusion Matrix")
plt.show()
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1],'--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LMDA ROC')
plt.legend()
plt.show()
print(f"LMDA ROC AUC: {roc_auc:.4f}")

# GCN
print("\nTraining GCN...")
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

train_graphs = [create_graph(arr, label) for arr, label in zip(X_train_bal, y_train_bal)]
test_graphs = [create_graph(arr, label) for arr, label in zip(X_test, y_test)]

train_loader_gcn = GeoDataLoader(train_graphs, batch_size=16, shuffle=True)
test_loader_gcn = GeoDataLoader(test_graphs, batch_size=16, shuffle=False)

gcn = GCN(node_feat_dim=n_freq, num_classes=2).to(device)
optimizer = torch.optim.Adam(gcn.parameters(), lr=1e-4)

for epoch in range(epochs):
    gcn.train()
    for batch in train_loader_gcn:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = gcn(batch)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()

gcn.eval()
all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for batch in test_loader_gcn:
        batch = batch.to(device)
        out = gcn(batch)
        prob = torch.softmax(out, dim=1)[:,1]
        pred = out.argmax(dim=1)
        all_probs.extend(prob.cpu().numpy())
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch.y.view(-1).cpu().numpy())

print("GCN Results:")
cm = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(cm, display_labels=["Stay", "Leave"]).plot(cmap=plt.cm.Blues)
plt.title("GCN Confusion Matrix")
plt.show()
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1],'--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('GCN ROC')
plt.legend()
plt.show()
print(f"GCN ROC AUC: {roc_auc:.4f}")
