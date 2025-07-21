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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 1. Simplified Anxiety Stage Classification (3 levels)
def create_simplified_stages(trait_df, score_col='Trait_anx'):
    bins = [25, 40, 50, 75]  # Low:25-39, Medium:40-49, High:50-74
    labels = ['Low(25-39)', 'Medium(40-49)', 'High(50-74)']
    trait_df['anxiety_stage'] = pd.cut(trait_df[score_col], bins=bins, labels=labels, right=False)
    return trait_df

# 2. EEG Data Loading
def extract_subid(fname):
    match = re.match(r'p(\d+)_', fname)
    return int(match.group(1)) if match else None

def load_all_epochs(folder, subid_to_stage, epoch_size=1000, overlap=0.5):
    all_epochs = []
    for fname in os.listdir(folder):
        if fname.endswith('.csv'):
            subid = extract_subid(fname)
            stage = subid_to_stage.get(subid, None)
            if stage is None: continue
            df = pd.read_csv(os.path.join(folder, fname))
            data = df.select_dtypes(include=[np.number]).values.T
            step = int(epoch_size * (1 - overlap))
            for start in range(0, data.shape[1] - epoch_size + 1, step):
                epoch = data[:, start:start+epoch_size]
                all_epochs.append((epoch, stage))
    return all_epochs

# 3. Model Definitions (Fixed)
class EEGNet(nn.Module):
    """EEGNet with Adaptive Pooling"""
    def __init__(self, chans=22, samples=1000, num_classes=3):
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
    """LMDA with Adaptive Pooling and Corrected Input Handling"""
    def __init__(self, chans=22, samples=1000, num_classes=3, depth=9):
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
        # Input shape: [batch, depth, 1, channels, time]
        # Remove the unnecessary dimension: [batch, depth, channels, time]
        x = x.squeeze(2)
        
        # Original LMDA operations
        x = torch.einsum('bdcw,hdc->bhcw', x, self.channel_weight)
        x = self.time_conv(x)
        x = self.chanel_conv(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class GCN(nn.Module):
    """GCN for 3-class classification"""
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

# 4. Training Pipeline
def train_model(model, train_loader, test_loader, num_epochs=20, model_type='eegnet'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training
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
    
    # Evaluation
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

# 5. Memory-Efficient Dataset for LMDA
class LMDADataset(torch.utils.data.Dataset):
    def __init__(self, X, y, depth=9):
        self.X = X
        self.y = y
        self.depth = depth
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        epoch = self.X[idx]  # [channels, time]
        # Create depth dimension without adding extra dimension
        epoch_depth = np.repeat(epoch[np.newaxis, :, :], self.depth, axis=0)
        return torch.tensor(epoch_depth, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# 6. Main Execution
def main():
    # Configuration
    low_dir = 'Anx/env2/Low'
    high_dir = 'Anx/env2/High'
    trait_csv = 'Downloads/stai_scores_subjectwise.csv'
    epoch_size = 1000
    overlap = 0.5
    
    # Load and process trait data
    trait_df = pd.read_csv(trait_csv)
    trait_df = create_simplified_stages(trait_df)
    subid_to_stage = trait_df.set_index('subid')['anxiety_stage'].to_dict()
    stage_to_idx = {'Low(25-39)':0, 'Medium(40-49)':1, 'High(50-74)':2}
    
    # Load EEG data
    data_low = load_all_epochs(low_dir, subid_to_stage, epoch_size, overlap)
    data_high = load_all_epochs(high_dir, subid_to_stage, epoch_size, overlap)
    all_data = data_low + data_high
    print(f"Loaded {len(all_data)} epochs")
    
    # Prepare datasets
    X = [arr for arr, _ in all_data]
    y = [stage_to_idx[stage] for _, stage in all_data]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model 1: EEGNet
    print("\nTraining EEGNet...")
    # Prepare EEGNet inputs: [batch, 1, channels, time]
    X_train_eegnet = np.array(X_train).transpose(0, 2, 1)[:, np.newaxis, :, :]
    X_test_eegnet = np.array(X_test).transpose(0, 2, 1)[:, np.newaxis, :, :]
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train_eegnet, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test_eegnet, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    
    eegnet = EEGNet(chans=X_train_eegnet.shape[2], samples=X_train_eegnet.shape[3])
    eegnet_labels, eegnet_preds = train_model(eegnet, train_loader, test_loader, model_type='eegnet')
    print("EEGNet Results:")
    print(classification_report(eegnet_labels, eegnet_preds, target_names=list(stage_to_idx.keys())))
    
    # Model 2: LMDA
    print("\nTraining LMDA...")
    # Create memory-efficient datasets
    train_dataset = LMDADataset(X_train, y_train)
    test_dataset = LMDADataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)
    
    # Auto-detect input dimensions
    chans = len(X_train[0])
    samples = len(X_train[0][0])
    lmda = LMDA(chans=chans, samples=samples)
    lmda_labels, lmda_preds = train_model(lmda, train_loader, test_loader, model_type='lmda')
    print("LMDA Results:")
    print(classification_report(lmda_labels, lmda_preds, target_names=list(stage_to_idx.keys())))
    
    # Model 3: GCN
    print("\nTraining GCN...")
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
    
    # Create graphs for all data
    graphs = [create_graph(arr, label) for arr, label in zip(X_train + X_test, y_train + y_test)]
    
    # Split into train/test
    train_graphs = [graphs[i] for i in range(len(X_train))]
    test_graphs = [graphs[i] for i in range(len(X_train), len(graphs))]
    
    train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=8)
    
    gcn = GCN(node_feat_dim=epoch_size, num_classes=3)
    gcn_labels, gcn_preds = train_model(gcn, train_loader, test_loader, model_type='gcn')
    print("GCN Results:")
    print(classification_report(gcn_labels, gcn_preds, target_names=list(stage_to_idx.keys())))

if __name__ == "__main__":
    main()
