import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ------------ 1. Frequency Domain Epoch Extraction ------------
def create_simplified_stages(trait_df, score_col='Trait_anx'):
    bins = [25, 40, 50, 75]
    labels = ['Low(25-39)', 'Medium(40-49)', 'High(50-74)']
    trait_df['anxiety_stage'] = pd.cut(trait_df[score_col], bins=bins, labels=labels, right=False)
    return trait_df

def extract_subid(fname):
    match = re.match(r'p(\d+)_', fname)
    return int(match.group(1)) if match else None

def extract_frequency_epochs(folder, subid_to_stage, epoch_size=1000, overlap=0.5, freq_bins=100):
    all_epochs = []
    for fname in os.listdir(folder):
        if fname.endswith('.csv'):
            subid = extract_subid(fname)
            stage = subid_to_stage.get(subid, None)
            if stage is None: continue
            df = pd.read_csv(os.path.join(folder, fname))
            data = df.select_dtypes(include=[np.number]).values.T  # (channels, time)
            step = int(epoch_size * (1 - overlap))
            for start in range(0, data.shape[1] - epoch_size + 1, step):
                epoch = data[:, start:start+epoch_size]
                fft_vals = np.abs(np.fft.rfft(epoch, axis=1))
                freq_feat = fft_vals[:, :freq_bins]  # (channels, freq_bins)
                all_epochs.append((freq_feat, stage))
    return all_epochs

class EEGNet(nn.Module):
    def __init__(self, chans=22, freq_bins=100, num_classes=3):
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
    def __init__(self, chans=22, freq_bins=100, num_classes=3, depth=9):
        super().__init__()
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, chans))
        nn.init.xavier_uniform_(self.channel_weight.data)
        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, 24, (1, min(75, freq_bins)), padding=(0, min(75 // 2, freq_bins // 2))),
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
        x = x.squeeze(2)
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

class LMDADataset(torch.utils.data.Dataset):
    def __init__(self, X, y, depth=9):
        self.X = X
        self.y = y
        self.depth = depth
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        epoch = self.X[idx]
        epoch_depth = np.repeat(epoch[np.newaxis, :, :], self.depth, axis=0)
        return torch.tensor(epoch_depth, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

def main():
    global X_train, y_train, X_test, y_test  # Expose these for visualization!

    low_dir = 'Anx/env2/Low'
    high_dir = 'Anx/env2/High'
    trait_csv = 'Downloads/stai_scores_subjectwise.csv'
    epoch_size = 1000
    overlap = 0.5
    freq_bins = 100

    trait_df = pd.read_csv(trait_csv)
    trait_df = create_simplified_stages(trait_df)
    subid_to_stage = trait_df.set_index('subid')['anxiety_stage'].to_dict()
    stage_to_idx = {'Low(25-39)':0, 'Medium(40-49)':1, 'High(50-74)':2}

    data_low = extract_frequency_epochs(low_dir, subid_to_stage, epoch_size, overlap, freq_bins)
    data_high = extract_frequency_epochs(high_dir, subid_to_stage, epoch_size, overlap, freq_bins)
    all_data = data_low + data_high
    print(f"Loaded {len(all_data)} frequency-domain epochs")

    X = [arr for arr, _ in all_data]
    y = [stage_to_idx[stage] for _, stage in all_data]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ----- EEGNet -----
    print("\nTraining EEGNet on Frequency Features...")
    X_train_eegnet = np.array(X_train)[:, np.newaxis, :, :]
    X_test_eegnet = np.array(X_test)[:, np.newaxis, :, :]
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_eegnet, dtype=torch.float32),
                                                  torch.tensor(y_train, dtype=torch.long))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test_eegnet, dtype=torch.float32),
                                                 torch.tensor(y_test, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    eegnet = EEGNet(chans=X_train_eegnet.shape[2], freq_bins=X_train_eegnet.shape[3])
    eegnet_labels, eegnet_preds = train_model(eegnet, train_loader, test_loader, model_type='eegnet')
    print("EEGNet Results:")
    print(classification_report(eegnet_labels, eegnet_preds, target_names=list(stage_to_idx.keys())))
    cm = confusion_matrix(eegnet_labels, eegnet_preds)
    ConfusionMatrixDisplay(cm, display_labels=list(stage_to_idx.keys())).plot(cmap='Blues')
    plt.title("EEGNet Confusion Matrix (Freq)")
    plt.show()

    # ----- LMDA -----
    print("\nTraining LMDA on Frequency Features...")
    train_dataset = LMDADataset(X_train, y_train)
    test_dataset = LMDADataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)
    chans = len(X_train[0])
    samples = len(X_train[0][0])
    lmda = LMDA(chans=chans, freq_bins=samples)
    lmda_labels, lmda_preds = train_model(lmda, train_loader, test_loader, model_type='lmda')
    print("LMDA Results:")
    print(classification_report(lmda_labels, lmda_preds, target_names=list(stage_to_idx.keys())))
    cm = confusion_matrix(lmda_labels, lmda_preds)
    ConfusionMatrixDisplay(cm, display_labels=list(stage_to_idx.keys())).plot(cmap='Blues')
    plt.title("LMDA Confusion Matrix (Freq)")
    plt.show()

    # ----- GCN -----
    print("\nTraining GCN on Frequency Features...")
    def create_graph(epoch_arr, label):
        n_channels, _ = epoch_arr.shape
        node_feats = torch.tensor(epoch_arr, dtype=torch.float32)
        corr = np.corrcoef(epoch_arr)
        edge_index = []
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                if abs(corr[i, j]) > 0.7:
                    edge_index.extend([[i, j], [j, i]])
        if not edge_index:
            edge_index = [[i, i] for i in range(n_channels)]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return Data(x=node_feats, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
    graphs = [create_graph(arr, label) for arr, label in zip(X_train + X_test, y_train + y_test)]
    train_graphs = [graphs[i] for i in range(len(X_train))]
    test_graphs = [graphs[i] for i in range(len(X_train), len(graphs))]
    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=8)
    gcn = GCN(node_feat_dim=freq_bins, num_classes=3)
    gcn_labels, gcn_preds = train_model(gcn, train_loader, test_loader, model_type='gcn')
    print("GCN Results:")
    print(classification_report(gcn_labels, gcn_preds, target_names=list(stage_to_idx.keys())))
    cm = confusion_matrix(gcn_labels, gcn_preds)
    ConfusionMatrixDisplay(cm, display_labels=list(stage_to_idx.keys())).plot(cmap='Blues')
    plt.title("GCN Confusion Matrix (Freq)")
    plt.show()
global lmda, gcn, X_train

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np

sample_idx = 0  # change as needed
X_train_np = np.array(X_train)
sample_features = X_train_np[sample_idx]
label = y_train[sample_idx]

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

for ch in range(sample_features.shape[0]):
    axs[0].plot(sample_features[ch], label=f'Ch {ch+1}')
axs[0].set_title(f'Line Plot (Sample {sample_idx}, Label: {label})')
axs[0].set_xlabel('Frequency Bin')
axs[0].set_ylabel('Magnitude')
axs[0].legend(fontsize='x-small', ncol=2, bbox_to_anchor=(1.05, 1), loc='upper left')

im = axs[1].imshow(sample_features, aspect='auto', cmap='viridis')
axs[1].set_title('Feature Heatmap')
axs[1].set_xlabel('Frequency Bin')
axs[1].set_ylabel('Channel')
fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

mean_per_channel = sample_features.mean(axis=1)
axs[2].bar(range(1, len(mean_per_channel)+1), mean_per_channel)
axs[2].set_title('Mean Magnitude per Channel')
axs[2].set_xlabel('Channel')
axs[2].set_ylabel('Mean Magnitude')

plt.tight_layout()
plt.show()

import torch
import torch.nn.functional as F
import numpy as np

# Instantiate EEGNet (adjust channels and freq_bins as needed)
eegnet = EEGNet(chans=22, freq_bins=100)

# Use a sample from your training data
sample_input = torch.tensor(np.array(X_train)[0][np.newaxis, np.newaxis, :, :], dtype=torch.float32)

with torch.no_grad():
    x = sample_input
    print("EEGNet conv1:", eegnet.conv1(x).shape)
    x = F.elu(eegnet.bn1(eegnet.conv1(x)))
    print("EEGNet bn1 + elu:", x.shape)
    x = F.elu(eegnet.bn2(eegnet.depthwise(x)))
    print("EEGNet depthwise + bn2 + elu:", x.shape)
    x = F.elu(eegnet.bn3(eegnet.conv2(x)))
    print("EEGNet conv2 + bn3 + elu:", x.shape)
    x = eegnet.adaptive_pool(x)
    print("EEGNet adaptive_pool:", x.shape)
    x = x.view(x.size(0), -1)
    print("EEGNet flatten:", x.shape)
    out = eegnet.classifier(x)
    print("EEGNet classifier:", out.shape)

import torch
import torch.nn.functional as F
import numpy as np

# Print the EEGNet model structure (all layers)
print("EEGNet Model Structure:\n")
print(eegnet)
print("\n" + "="*40 + "\n")

# Print output shape at each layer for a sample input
sample_input = torch.tensor(np.array(X_train)[0][np.newaxis, np.newaxis, :, :], dtype=torch.float32)

with torch.no_grad():
    x = sample_input
    print("Input shape:", x.shape)
    x = eegnet.conv1(x)
    print("After conv1:", x.shape)
    x = eegnet.bn1(x)
    print("After bn1:", x.shape)
    x = F.elu(x)
    print("After ELU 1:", x.shape)
    x = eegnet.depthwise(x)
    print("After depthwise:", x.shape)
    x = eegnet.bn2(x)
    print("After bn2:", x.shape)
    x = F.elu(x)
    print("After ELU 2:", x.shape)
    x = eegnet.conv2(x)
    print("After conv2:", x.shape)
    x = eegnet.bn3(x)
    print("After bn3:", x.shape)
    x = F.elu(x)
    print("After ELU 3:", x.shape)
    x = eegnet.adaptive_pool(x)
    print("After adaptive_pool:", x.shape)
    x = x.view(x.size(0), -1)
    print("After flatten:", x.shape)
    out = eegnet.classifier(x)
    print("After classifier (output):", out.shape)

# Re-instantiate LMDA and GCN using the same parameters as your main code
chans = len(X_train[0])
samples = len(X_train[0][0])
freq_bins = samples

lmda = LMDA(chans=chans, freq_bins=freq_bins)
gcn = GCN(node_feat_dim=freq_bins, num_classes=3)

import torch
import torch.nn.functional as F
import numpy as np

print("LMDA Model Structure:\n")
print(lmda)
print("\n" + "="*40 + "\n")

depth = lmda.channel_weight.shape[0]
sample_input_lmda = torch.tensor(np.repeat(np.array(X_train)[0][np.newaxis, :, :], depth, axis=0)[np.newaxis], dtype=torch.float32)

with torch.no_grad():
    x = sample_input_lmda
    print("LMDA Input shape:", x.shape)
    x = x.squeeze(2)
    print("After squeeze(2):", x.shape)
    x = torch.einsum('bdcw,hdc->bhcw', x, lmda.channel_weight)
    print("After channel_weight einsum:", x.shape)
    x = lmda.time_conv(x)
    print("After time_conv:", x.shape)
    x = lmda.chanel_conv(x)
    print("After chanel_conv:", x.shape)
    x = lmda.adaptive_pool(x)
    print("After adaptive_pool:", x.shape)
    x = x.view(x.size(0), -1)
    print("After flatten:", x.shape)
    out = lmda.classifier(x)
    print("After classifier (output):", out.shape)

print("\n" + "="*60 + "\n")

print("GCN Model Structure:\n")
print(gcn)
print("\n" + "="*40 + "\n")

sample_epoch = np.array(X_train)[0]
n_channels, freq_bins = sample_epoch.shape
node_feats = torch.tensor(sample_epoch, dtype=torch.float32)
corr = np.corrcoef(sample_epoch)
edge_index = []
for i in range(n_channels):
    for j in range(i+1, n_channels):
        if abs(corr[i, j]) > 0.7:
            edge_index.extend([[i, j], [j, i]])
if not edge_index:
    edge_index = [[i, i] for i in range(n_channels)]
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

batch = torch.zeros(node_feats.size(0), dtype=torch.long)

print("GCN layer-wise output shapes:")
with torch.no_grad():
    x = node_feats
    print("Input node features:", x.shape)
    print("Edge index shape:", edge_index.shape)
    x = gcn.conv1(x, edge_index)
    print("After conv1:", x.shape)
    x = F.relu(x)
    print("After ReLU 1:", x.shape)
    x = gcn.conv2(x, edge_index)
    print("After conv2:", x.shape)
    x = F.relu(x)
    print("After ReLU 2:", x.shape)
    from torch_geometric.nn import global_mean_pool
    x = global_mean_pool(x, batch)
    print("After global_mean_pool:", x.shape)
    out = gcn.classifier(x)
    print("After classifier (output):", out.shape)
