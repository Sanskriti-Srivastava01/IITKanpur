import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# CONFIG
USE_FREQUENCY_DOMAIN = False # Set to True for FFT input, False for time domain
EPOCH_SIZE = 1000
OVERLAP = 0.5
FREQ_BINS = 100       # Only used if USE_FREQUENCY_DOMAIN=True
eeg_dirs = {'Low': 'Anx/env1/Low', 'High': 'Anx/env1/High'}  # Add more classes as needed

def extract_epochs_from_csv(folder, label, epoch_size=1000, overlap=0.5, freq_bins=100, use_freq=False):
    epochs = []
    for fname in os.listdir(folder):
        if fname.endswith('.csv'):
            fpath = os.path.join(folder, fname)
            df = pd.read_csv(fpath)
            data = df.select_dtypes(include=[np.number]).values.T  # (channels, time)
            step = int(epoch_size * (1 - overlap))
            for start in range(0, data.shape[1] - epoch_size + 1, step):
                epoch = data[:, start:start+epoch_size]  # (channels, epoch_size)
                if use_freq:
                    # Frequency domain: FFT and keep freq_bins
                    fft_vals = np.abs(np.fft.rfft(epoch, axis=1))
                    feat = fft_vals[:, :freq_bins]  # (channels, freq_bins)
                    # For LSTM input: (freq_bins, channels)
                    feat = feat.T
                else:
                    # Time domain: (epoch_size, channels)
                    feat = epoch.T
                epochs.append((feat, label))
    return epochs

# 1. Load all labeled epochs
all_data = []
for label_name, folder in eeg_dirs.items():
    label = list(eeg_dirs.keys()).index(label_name)  # Encode labels as integers
    all_data += extract_epochs_from_csv(folder, label, EPOCH_SIZE, OVERLAP, FREQ_BINS, use_freq=USE_FREQUENCY_DOMAIN)

print(f"Number of epochs: {len(all_data)}")

X = [arr for arr, _ in all_data]
y = [label for _, label in all_data]
X = np.array(X)
y = np.array(y)
print(X.shape, y.shape)

# 2. Standardize features
n_samples, n_steps, n_features = X.shape
X_reshaped = X.reshape(-1, n_features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)
X_scaled = X_scaled.reshape(n_samples, n_steps, n_features)

# One-hot encode labels for classification
num_classes = len(np.unique(y))
y_cat = tf.keras.utils.to_categorical(y, num_classes=num_classes)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=0.2, random_state=42, stratify=y_cat)

# 4. Model Definitions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout

def build_lstm(input_shape, num_classes):
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=input_shape),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_bilstm(input_shape, num_classes):
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=False), input_shape=input_shape),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 5. Training, Evaluation, Performance Table

# Train LSTM
lstm_model = build_lstm((n_steps, n_features), num_classes)
history_lstm = lstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)
lstm_eval = lstm_model.evaluate(X_test, y_test, verbose=0)
y_pred_lstm = np.argmax(lstm_model.predict(X_test), axis=1)
y_true_lstm = np.argmax(y_test, axis=1)

print("LSTM Classification Report:")
print(classification_report(y_true_lstm, y_pred_lstm))
cm1 = confusion_matrix(y_true_lstm, y_pred_lstm)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1)
disp1.plot(cmap='Blues')
plt.title('LSTM Confusion Matrix')
plt.show()

# Train Bi-LSTM
bilstm_model = build_bilstm((n_steps, n_features), num_classes)
history_bilstm = bilstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)
bilstm_eval = bilstm_model.evaluate(X_test, y_test, verbose=0)
y_pred_bilstm = np.argmax(bilstm_model.predict(X_test), axis=1)
y_true_bilstm = np.argmax(y_test, axis=1)

print("Bi-LSTM Classification Report:")
print(classification_report(y_true_bilstm, y_pred_bilstm))
cm2 = confusion_matrix(y_true_bilstm, y_pred_bilstm)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
disp2.plot(cmap='Blues')
plt.title('Bi-LSTM Confusion Matrix')
plt.show()

# Table summarizing results
import pandas as pd
results = {
    'Model': ['LSTM', 'Bi-LSTM'],
    'Test Accuracy': [lstm_eval[1], bilstm_eval[1]],
    'Test Loss': [lstm_eval, bilstm_eval],
}
summary_df = pd.DataFrame(results)
print(summary_df.to_markdown(index=False))
