import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Convert EEG epochs to frequency domain
def epoch_to_frequency_features(epoch, fs=1000, freq_bins=100):
    n_channels, n_time = epoch.shape
    fft_vals = np.fft.rfft(epoch, axis=1)
    power = np.abs(fft_vals)
    power = power[:, :freq_bins]
    return power

# 2. Sliding Window Epoch Extraction for All Files
def load_all_epochs_freq(folder, label, epoch_size=1000, overlap=0.5, fs=1000, freq_bins=100):
    all_epochs = []
    for fname in os.listdir(folder):
        if fname.endswith('.csv'):
            file_path = os.path.join(folder, fname)
            df = pd.read_csv(file_path)
            data = df.select_dtypes(include=[np.number]).values.T  # (channels, time)
            step = int(epoch_size * (1 - overlap))
            for start in range(0, data.shape[1] - epoch_size + 1, step):
                epoch = data[:, start:start+epoch_size]
                freq_features = epoch_to_frequency_features(epoch, fs=fs, freq_bins=freq_bins)
                all_epochs.append((freq_features, label))
    return all_epochs

# 3. Model Definitions
class EEGNet(nn.Module):
    def __init__(self, chans=22, freq_bins=100, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.depthwise = nn.Conv2d(16, 32, (chans, 1), groups=16)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (1, 16), padding='same')
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
    def __init__(self, chans=22, freq_bins=100, num_classes=3, depth=9):
        super().__init__()
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, chans))
        nn.init.xavier_uniform_(self.channel_weight.data)
        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, 24, (1, 75), padding=(0, 37)),
            nn.BatchNorm2d(24),
            nn.GELU(),
        )
        self.chanel_conv = nn.Sequential(
            nn.Conv2d(24, 24, (chans, 1), groups=24),
            nn.BatchNorm2d(24),
            nn.GELU(),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(24, num_classes)
    def forward(self, x):
        x = x.squeeze(2)  # Remove extra dim if present
        x = torch.einsum('bdcw,hdc->bhcw', x, self.channel_weight)
        x = self.time_conv(x)
        x = self.chanel_conv(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class GCN(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim=32, num_classes=3):
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

# 4. Training and Evaluation
def train_model(model, train_loader, test_loader, num_epochs=20, model_type='eegnet'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            if model_type == 'gcn':
                batch = batch.to(device)
                outputs = model(batch)
                y = batch.y.view(-1)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                outputs = model(x)
            optimizer.zero_grad()
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}: Loss {total_loss/len(train_loader):.4f}')
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            if model_type == 'gcn':
                batch = batch.to(device)
                outputs = model(batch)
                y = batch.y.view(-1)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    return all_labels, all_preds

# 5. Main Pipeline
def main():
    low_dir = 'Anx/env1/Low'
    high_dir = 'Anx/env1/High'
    epoch_size = 1000
    overlap = 0.5
    fs = 1000
    freq_bins = 100

    # Load all epochs and convert to frequency domain
    data_low = load_all_epochs_freq(low_dir, label=0, epoch_size=epoch_size, overlap=overlap, fs=fs, freq_bins=freq_bins)
    data_high = load_all_epochs_freq(high_dir, label=1, epoch_size=epoch_size, overlap=overlap, fs=fs, freq_bins=freq_bins)
    all_data = data_low + data_high
    print(f"Loaded {len(all_data)} frequency domain epochs.")

    # Prepare data for EEGNet and LMDA
    X = [arr for arr, _ in all_data]
    y = [label for _, label in all_data]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # EEGNet inputs: [batch, 1, channels, freq_bins]
    X_train_eegnet = np.array(X_train)[:, np.newaxis, :, :]
    X_test_eegnet = np.array(X_test)[:, np.newaxis, :, :]

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_eegnet, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test_eegnet, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

    eegnet = EEGNet(chans=X_train_eegnet.shape[2], freq_bins=X_train_eegnet.shape[3])
    eegnet_labels, eegnet_preds = train_model(eegnet, train_loader, test_loader, model_type='eegnet')
    print("EEGNet Results:")
    print(classification_report(eegnet_labels, eegnet_preds))

    # LMDA inputs: [batch, depth, channels, freq_bins]
    X_train_lmda = np.array(X_train)[:, np.newaxis, :, :]
    X_train_lmda = np.repeat(X_train_lmda, 9, axis=1)
    X_test_lmda = np.array(X_test)[:, np.newaxis, :, :]
    X_test_lmda = np.repeat(X_test_lmda, 9, axis=1)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_lmda, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test_lmda, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    lmda = LMDA(chans=X_train_lmda.shape[2], freq_bins=X_train_lmda.shape[3])
    lmda_labels, lmda_preds = train_model(lmda, train_loader, test_loader, model_type='lmda')
    print("LMDA Results:")
    print(classification_report(lmda_labels, lmda_preds))

    # GCN inputs: create graphs with node features as freq bins
    def create_graph_freq(epoch_arr, label, corr_threshold=0.7):
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

    graphs = [create_graph_freq(arr, label) for arr, label in zip(X_train + X_test, y_train + y_test)]
    train_graphs = [graphs[i] for i in range(len(X_train))]
    test_graphs = [graphs[i] for i in range(len(X_train), len(graphs))]

    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=8)

    gcn = GCN(node_feat_dim=freq_bins, num_classes=3)
    gcn_labels, gcn_preds = train_model(gcn, train_loader, test_loader, model_type='gcn')
    print("GCN Results:")
    print(classification_report(gcn_labels, gcn_preds))

if __name__ == "__main__":
    main()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# After EEGNet results
print("EEGNet Results:")
print(classification_report(eegnet_labels, eegnet_preds))
cm = confusion_matrix(eegnet_labels, eegnet_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'Medium', 'High'])
disp.plot(cmap='Blues')
plt.title('EEGNet Confusion Matrix')
plt.show()

# After LMDA results
print("LMDA Results:")
print(classification_report(lmda_labels, lmda_preds))
cm = confusion_matrix(lmda_labels, lmda_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'Medium', 'High'])
disp.plot(cmap='Blues')
plt.title('LMDA Confusion Matrix')
plt.show()

# After GCN results
print("GCN Results:")
print(classification_report(gcn_labels, gcn_preds))
cm = confusion_matrix(gcn_labels, gcn_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'Medium', 'High'])
disp.plot(cmap='Blues')
plt.title('GCN Confusion Matrix')
plt.show()
