import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

def create_custom_anxiety_stages(trait_df, score_col='Trait_anx'):
    bins = [25, 35, 45, 55, 65, 75, 85]  # 25-34, 35-44, ..., 75-84
    labels = [f'{bins[i]}-{bins[i+1]-1}' for i in range(len(bins)-1)]
    trait_df['anxiety_stage'] = pd.cut(trait_df[score_col], bins=bins, labels=labels, right=False)
    return trait_df

# 2. Extract subject ID from filename
def extract_subid(fname):
    match = re.match(r'p(\d+)_', fname)
    return int(match.group(1)) if match else None

# 3. Load all epochs with stage labels
def load_all_epochs_with_stage(folder, subid_to_stage, epoch_size=1000, overlap=0.5):
    all_epochs = []
    for fname in os.listdir(folder):
        if fname.endswith('.csv'):
            subid = extract_subid(fname)
            stage = subid_to_stage.get(subid, None)
            if stage is None or pd.isna(stage):
                continue
            file_path = os.path.join(folder, fname)
            df = pd.read_csv(file_path)
            data = df.select_dtypes(include=[np.number]).values.T
            step = int(epoch_size * (1 - overlap))
            for start in range(0, data.shape[1] - epoch_size + 1, step):
                epoch = data[:, start:start+epoch_size]
                all_epochs.append((epoch, stage))
    return all_epochs

# 4. Convert epoch to PyTorch Geometric Data graph
def epoch_to_graph(epoch_arr, label, stage_to_idx, corr_threshold=0.7):
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
    label_idx = stage_to_idx[label]
    return Data(x=node_feats, edge_index=edge_index, y=torch.tensor([label_idx], dtype=torch.long))

# 5. LMDA Model Definition (as per your provided code)
class LMDA(nn.Module):
    def __init__(self, chans=22, samples=1000, num_classes=13, depth=9, kernel=75, 
                 channel_depth1=24, channel_depth2=24):
        super(LMDA, self).__init__()
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, chans), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)
        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, (1, kernel), padding=(0, kernel//2), bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
            nn.AvgPool2d((1, 4))
        )
        self.chanel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, (chans, 1), groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
            nn.AvgPool2d((1, 8))
        )
        with torch.no_grad():
            dummy = torch.ones(1, depth, chans, samples)
            out = torch.einsum('bdcw,hdc->bhcw', dummy, self.channel_weight)
            out = self.time_conv(out)
            out = self.chanel_conv(out)
            self.fc_input = out.numel() // out.shape[0]
        self.classifier = nn.Linear(self.fc_input, num_classes)
    
    def EEGDepthAttention(self, x):
        N, C, H, W = x.size()
        k = 7
        adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k//2, 0), bias=True).to(x.device)
        softmax = nn.Softmax(dim=-2)
        x_pool = adaptive_pool(x)
        x_transpose = x_pool.transpose(1, 2)
        y = conv(x_transpose)
        y = softmax(y)
        y = y.transpose(1, 2)
        return y * C * x
    
    def forward(self, x):
        x = torch.einsum('bdcw,hdc->bhcw', x, self.channel_weight)
        x = self.time_conv(x)
        x = self.EEGDepthAttention(x)
        x = self.chanel_conv(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# 6. Main pipeline
def main(low_dir, high_dir, trait_csv, epoch_size=1000, overlap=0.5, corr_threshold=0.7):
    # Load and process trait data
    trait_df = pd.read_csv(trait_csv)
    trait_df = create_anxiety_stages(trait_df)
    subid_to_stage = trait_df.set_index('subid')['anxiety_stage'].to_dict()
    
    # Load EEG epochs
    data_low = load_all_epochs_with_stage(low_dir, subid_to_stage, epoch_size, overlap)
    data_high = load_all_epochs_with_stage(high_dir, subid_to_stage, epoch_size, overlap)
    all_epochs = data_low + data_high
    print(f"Loaded {len(all_epochs)} epochs with anxiety stages.")
    
    # Prepare stages and labels
    all_stages = sorted(trait_df['anxiety_stage'].dropna().unique())
    stage_to_idx = {stage: idx for idx, stage in enumerate(all_stages)}
    num_classes = len(all_stages)
    
    # Convert to graphs
    graphs = []
    for epoch, stage in all_epochs:
        graphs.append(epoch_to_graph(epoch, stage, stage_to_idx, corr_threshold))
    
    # Train/test split
    labels = [g.y.item() for g in graphs]
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42, stratify=labels)
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=16)
    
    # Initialize model and training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LMDA(chans=graphs[0].x.shape[0],  # Number of channels
                 samples=epoch_size, 
                 num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            
            # Prepare input for LMDA: [batch_size, depth, chans, samples]
            # Reshape from [total_nodes, time] to [batch_size, channels, time]
            batch_size = batch.num_graphs
            n_channels = batch.x.shape[0] // batch_size
            x = batch.x.view(batch_size, n_channels, -1)  # [batch_size, channels, time]
            
            # Add depth dimension by repeating
            x = x.unsqueeze(1).repeat(1, 9, 1, 1)  # [batch_size, depth, channels, time]
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(train_loader):.4f}')
    
    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            batch_size = batch.num_graphs
            n_channels = batch.x.shape[0] // batch_size
            x = batch.x.view(batch_size, n_channels, -1)
            x = x.unsqueeze(1).repeat(1, 9, 1, 1)
            
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.view(-1).cpu().numpy())
    
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {acc:.4f}')
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(s) for s in all_stages])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()
    print(classification_report(all_labels, all_preds, target_names=[str(s) for s in all_stages]))

# Example usage:
main('Anx/env2/Low', 'Anx/env2/High', 'Downloads/stai_scores_subjectwise.csv')
