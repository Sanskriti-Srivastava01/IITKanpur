import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, BatchNormalization
from tensorflow.keras.layers import Activation, AveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import re

# ========== 1. EEGNet Model Definition ==========
def EEGNet(nb_classes, Chans=64, Samples=128, dropoutRate=0.5, 
           kernLength=64, F1=8, D=2, F2=16):
    input1 = Input(shape=(Chans, Samples, 1))

    # Block 1
    block1 = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D)(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    # Block 2
    block2 = SeparableConv2D(F2, (1, 16), padding='same', use_bias=False)(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)

    flatten = Flatten()(block2)
    dense = Dense(nb_classes)(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input1, outputs=softmax)

# ========== 2. Data Loading and Processing ==========
def load_subject_data(file_path):
    """Load and process a single subject's CSV file"""
    df = pd.read_csv(file_path)
    
    # Extract numeric data only
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Remove epoch column if exists
    if 'Epoch' in df_numeric.columns:
        df_numeric = df_numeric.drop(columns=['Epoch'])
    
    # Standardize data
    scaler = StandardScaler()
    return scaler.fit_transform(df_numeric.values)

def create_epochs(data, epoch_size=1000, overlap=0.5):
    """Create epochs from continuous data"""
    step = int(epoch_size * (1 - overlap))
    epochs = []
    for start in range(0, len(data) - epoch_size + 1, step):
        epochs.append(data[start:start+epoch_size])
    return np.array(epochs)

def load_all_data(low_dir, high_dir, epoch_size=1000):
    X, y, subjects = [], [], []
    
    # Process Low group
    for fname in os.listdir(low_dir):
        if fname.endswith('.csv'):
            file_path = os.path.join(low_dir, fname)
            data = load_subject_data(file_path)
            epochs = create_epochs(data, epoch_size)
            X.append(epochs)
            y.append(np.zeros(len(epochs)))  # Low label
            subjects.extend([fname.split('_')[0]] * len(epochs))
    
    # Process High group
    for fname in os.listdir(high_dir):
        if fname.endswith('.csv'):
            file_path = os.path.join(high_dir, fname)
            data = load_subject_data(file_path)
            epochs = create_epochs(data, epoch_size)
            X.append(epochs)
            y.append(np.ones(len(epochs)))  # High label
            subjects.extend([fname.split('_')[0]] * len(epochs))
    
    # Combine all epochs
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y, np.array(subjects)

# ========== 3. Subject-wise Split ==========
def subject_wise_split(X, y, subjects, test_size=0.2):
    unique_subjects = np.unique(subjects)
    train_subs, test_subs = train_test_split(unique_subjects, test_size=test_size, random_state=42)
    
    train_mask = np.isin(subjects, train_subs)
    test_mask = np.isin(subjects, test_subs)
    
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]

# ========== 4. Main Pipeline ==========
# Configuration
low_dir = 'Anx/env2/Low'
high_dir = 'Anx/env2/High'
epoch_size = 1000  # Adjust based on your sampling rate

# 1. Load and preprocess data
X, y, subjects = load_all_data(low_dir, high_dir, epoch_size)
print(f"Loaded data shape: {X.shape} (epochs, time_points, channels)")

# 2. Prepare EEGNet input (channels, time_points, 1)
X = np.transpose(X, (0, 2, 1))  # Convert to (epochs, channels, time_points)
X = X[..., np.newaxis]           # Add channel dimension
print(f"EEGNet input shape: {X.shape}")

# 3. Subject-wise split
X_train, X_test, y_train, y_test = subject_wise_split(X, y, subjects)

# 4. Prepare labels
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# 5. Model parameters
channels = X.shape[1]
time_points = X.shape[2]

# 6. Build EEGNet model
model = EEGNet(
    nb_classes=2,
    Chans=channels,
    Samples=time_points,
    kernLength=64,
    dropoutRate=0.5
)

# 7. Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# 8. Train with early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train_cat,
    batch_size=32,
    epochs=100,
    validation_data=(X_test, y_test_cat),
    callbacks=[early_stop],
    verbose=1
)

# 9. Evaluate
score = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Accuracy: {score[1]*100:.2f}%")
print(f"Test Loss: {score[0]:.4f}")
# Get aligned predictions and true labels
min_length = min(len(y_test), len(y_pred))
y_test_aligned = y_test[:min_length]
y_pred_aligned = y_pred[:min_length]

# Generate confusion matrix
cm = confusion_matrix(y_test_aligned, y_pred_aligned)

# Plot
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
