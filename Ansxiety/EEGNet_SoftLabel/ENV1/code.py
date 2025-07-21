import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, BatchNormalization
from tensorflow.keras.layers import Activation, AveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import re

# ========== 1. EEGNet Model Definition ==========
def EEGNet(nb_classes, Chans=64, Samples=128, dropoutRate=0.5, 
           kernLength=64, F1=8, D=2, F2=16, final_activation='softmax'):
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
    output = Activation(final_activation)(dense)

    return Model(inputs=input1, outputs=output)

# ========== 2. Data Loading and Processing ==========
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
            # Extract trait score from filename or external CSV as needed
            # Here, just use a placeholder (e.g., use a lookup table in practice)
            trait_scores.extend([0.0] * len(epochs))  # Replace with actual trait scores
    # Process High group
    for fname in os.listdir(high_dir):
        if fname.endswith('.csv'):
            file_path = os.path.join(high_dir, fname)
            data = load_subject_data(file_path)
            epochs = create_epochs(data, epoch_size)
            X.append(epochs)
            y.append(np.ones(len(epochs)))  # High label
            subjects.extend([fname.split('_')[0]] * len(epochs))
            trait_scores.extend([1.0] * len(epochs))  # Replace with actual trait scores
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y, np.array(subjects), np.array(trait_scores)

# ========== 3. Soft Label Function ==========
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

# ========== 4. Subject-wise Split ==========
def subject_wise_split(X, y, subjects, test_size=0.2):
    unique_subjects = np.unique(subjects)
    train_subs, test_subs = train_test_split(unique_subjects, test_size=test_size, random_state=42)
    train_mask = np.isin(subjects, train_subs)
    test_mask = np.isin(subjects, test_subs)
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]

# ========== 5. Main Pipeline ==========
# Configuration
low_dir = 'Anx/env1/Low'
high_dir = 'Anx/env1/High'
epoch_size = 1000  # Adjust as needed

# Choose mode: 'soft' for soft labels, 'reg' for regression
mode = 'soft'  # or 'reg'

# 1. Load and preprocess data
X, y, subjects, trait_scores = load_all_data(low_dir, high_dir, epoch_size)
print(f"Loaded data shape: {X.shape} (epochs, time_points, channels)")

X = np.transpose(X, (0, 2, 1))  # (epochs, channels, time_points)
X = X[..., np.newaxis]           # (epochs, channels, time_points, 1)
print(f"EEGNet input shape: {X.shape}")

# 2. Prepare labels
if mode == 'soft':
    # Replace trait_scores with your actual trait score array for each epoch
    median = np.median(trait_scores)
    score_range = trait_scores.max() - trait_scores.min()
    y_soft = create_soft_labels(trait_scores, median, score_range, buffer=0.2)
    y_train, y_test, y_train_soft, y_test_soft = train_test_split(y, y_soft, test_size=0.2, random_state=42, stratify=y)
elif mode == 'reg':
    # For regression, use trait_scores as continuous targets
    y_train, y_test, y_train_reg, y_test_reg = train_test_split(y, trait_scores, test_size=0.2, random_state=42, stratify=y)

# 3. Subject-wise split for X and y
X_train, X_test, sub_train, sub_test = train_test_split(X, subjects, test_size=0.2, random_state=42, stratify=y)

# 4. Build and compile model
channels = X.shape[1]
time_points = X.shape[2]

if mode == 'soft':
    model = EEGNet(nb_classes=2, Chans=channels, Samples=time_points, final_activation='softmax')
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    y_train_model, y_test_model = y_train_soft, y_test_soft
elif mode == 'reg':
    model = EEGNet(nb_classes=1, Chans=channels, Samples=time_points, final_activation='linear')
    model.compile(loss='mse', optimizer=Adam(0.001), metrics=['mae'])
    y_train_model, y_test_model = y_train_reg, y_test_reg

# 5. Train
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train_model,
    batch_size=32,
    epochs=100,
    validation_data=(X_test, y_test_model),
    callbacks=[early_stop],
    verbose=1
)

# 6. Evaluate
score = model.evaluate(X_test, y_test_model, verbose=0)
if mode == 'soft':
    print(f"\nTest Accuracy: {score[1]*100:.2f}%")
    print(f"Test Loss: {score[0]:.4f}")
elif mode == 'reg':
    print(f"\nTest MAE: {score[1]:.4f}")
    print(f"Test MSE: {score[0]:.4f}")
if mode == 'soft':
    # Generate predictions
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Convert soft labels to hard labels for true values
    y_true = np.argmax(y_test_model, axis=1)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low Anxiety', 'High Anxiety'], 
                yticklabels=['Low Anxiety', 'High Anxiety'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Anxiety Classification')
    plt.show()
    
    # Print classification report
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Low Anxiety', 'High Anxiety']))
