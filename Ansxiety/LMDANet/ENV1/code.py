import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ========== 1. LMDA Model Definition ==========
class LMDA(nn.Module):
    def __init__(self, chans=22, samples=1125, num_classes=2, depth=9, kernel=75, 
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

# ========== 2. Memory-Efficient Dataset ==========
class DepthReplicationDataset(Dataset):
    def __init__(self, X, y, depth=9):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.depth = depth
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x_sample = self.X[idx]  # (time_points, channels)
        # Replicate depth dimension on-the-fly
        x_depth = np.repeat(x_sample[np.newaxis, :, :], self.depth, axis=0)
        x_depth = np.transpose(x_depth, (0, 2, 1))  # (depth, channels, time_points)
        return x_depth, self.y[idx]

# ========== 3. Data Loading and Processing ==========
def load_subject_data(file_path):
    df = pd.read_csv(file_path)
    df_numeric = df.select_dtypes(include=[np.number])
    if 'Epoch' in df_numeric.columns:
        df_numeric = df_numeric.drop(columns=['Epoch'])
    scaler = StandardScaler()
    return scaler.fit_transform(df_numeric.values)

def create_epochs(data, epoch_size=1000, overlap=0.5):
    step = int(epoch_size * (1 - overlap))
    epochs = []
    for start in range(0, len(data) - epoch_size + 1, step):
        epochs.append(data[start:start+epoch_size])
    return np.array(epochs)

def load_all_data(low_dir, high_dir, epoch_size=1000):
    X, y, subjects, trait_scores = [], [], [], []
    # Process Low group
    for fname in os.listdir(low_dir):
        if fname.endswith('.csv'):
            file_path = os.path.join(low_dir, fname)
            data = load_subject_data(file_path)
            epochs = create_epochs(data, epoch_size)
            X.append(epochs)
            y.append(np.zeros(len(epochs)))  # Low label
            subjects.extend([fname.split('_')[0]] * len(epochs))
            trait_scores.extend([0.0] * len(epochs))  # Placeholder
    # Process High group
    for fname in os.listdir(high_dir):
        if fname.endswith('.csv'):
            file_path = os.path.join(high_dir, fname)
            data = load_subject_data(file_path)
            epochs = create_epochs(data, epoch_size)
            X.append(epochs)
            y.append(np.ones(len(epochs)))  # High label
            subjects.extend([fname.split('_')[0]] * len(epochs))
            trait_scores.extend([1.0] * len(epochs))  # Placeholder
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y, np.array(subjects), np.array(trait_scores)

# ========== 4. Soft Label Function ==========
def create_soft_labels(trait_scores, median, score_range, buffer=0.2):
    soft_labels = []
    for score in trait_scores:
        dist = (score - median) / (score_range/2)
        if dist < -buffer:
            soft_labels.append([1.0, 0.0])
        elif dist > buffer:
            soft_labels.append([0.0, 1.0])
        else:
            high_prob = 0.5 + 0.5 * dist / buffer
            low_prob = 1.0 - high_prob
            soft_labels.append([low_prob, high_prob])
    return np.array(soft_labels)

# ========== 5. Main Pipeline ==========
# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
low_dir = 'Anx/env1/Low'
high_dir = 'Anx/env1/High'
epoch_size = 1000
depth = 9
batch_size = 8
num_epochs = 100

# 1. Load and preprocess data
X, y, subjects, trait_scores = load_all_data(low_dir, high_dir, epoch_size)
print(f"Loaded data shape: {X.shape} (epochs, time_points, channels)")

# 2. Prepare labels
median = np.median(trait_scores)
score_range = trait_scores.max() - trait_scores.min()
y_soft = create_soft_labels(trait_scores, median, score_range, buffer=0.2)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_soft, test_size=0.2, random_state=42, stratify=y
)

# 4. Create datasets and loaders
train_dataset = DepthReplicationDataset(X_train, y_train, depth)
test_dataset = DepthReplicationDataset(X_test, y_test, depth)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# 5. Initialize LMDA model
chans = X_train.shape[2]
samples = X_train.shape[1]
model = LMDA(
    chans=chans,
    samples=samples,
    num_classes=2,
    depth=depth,
    kernel=75,
    channel_depth1=24,
    channel_depth2=24
).to(device)

# 6. Loss, optimizer, scheduler
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# 7. Training loop
best_acc = 0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # Validation
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            _, targets = torch.max(labels, 1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()
    avg_val_loss = val_loss / len(test_loader)
    scheduler.step(avg_val_loss)
    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1}/{num_epochs} | '
          f'Train Loss: {running_loss/len(train_loader):.4f} | '
          f'Val Loss: {avg_val_loss:.4f} | '
          f'Accuracy: {accuracy:.2f}%')
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), 'best_lmda_model.pth')

# 8. Evaluation
model.load_state_dict(torch.load('best_lmda_model.pth'))
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        _, targets = torch.max(labels, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# 9. Confusion Matrix and Classification Report
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low Anxiety', 'High Anxiety'], 
            yticklabels=['Low Anxiety', 'High Anxiety'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Anxiety Classification')
plt.show()

print("\nClassification Report:")
print(classification_report(all_targets, all_preds, target_names=['Low Anxiety', 'High Anxiety']))
