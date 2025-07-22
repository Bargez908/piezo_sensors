import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from itertools import combinations
import pywt
from scipy.signal import stft
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Conv1D, GlobalAveragePooling1D,
    LSTM, Input, LayerNormalization, MultiHeadAttention,
    Dropout, Flatten
)
from tensorflow.keras.optimizers import Adam

#os.system('cls')

# Load the sensor data
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define parameters for the data folders
model_type = "transformer"  # Choose between 'simple', 'cnn', 'rnn', or 'transformer'


macro_folder = "records_final"
test = "thumb_pressure"
test_n = 4
test_type = "pressure"  # "sliding" or "pressure"
finger="thumb"  # "index", "middle", "ring", "little", "thumb"

window_size = 200  # Example window size (adjust based on your data)
overlap = 0.9  # Example overlap (adjust based on your data)
sequence_length = 30  # number of windows per sequence for sequence-based models

number_splits = 5 #number of sequence splits for cross-validation

piezo_data_path = os.path.join(script_dir, "data", macro_folder, test, f'{finger}_concatenated.npy')
sensor_data_path = os.path.join(script_dir, "data", macro_folder, test, 'sensor_values_concatenated.npy')

sensor_data = np.load(sensor_data_path)
piezo_data = np.load(piezo_data_path)
sensor_data = sensor_data[:, 2]

# Define the type of feature extraction
feature_type = "wavelet"  # Choose between "wavelet" or "stft"

# Preprocess the data
assert piezo_data.shape[0] == sensor_data.shape[0], "Mismatch in the number of samples"

full_X = piezo_data
full_y = sensor_data

# Compute splits
part_len = len(full_X) // number_splits
X_parts = [full_X[i * part_len: (i + 1) * part_len] for i in range(number_splits)]
y_parts = [full_y[i * part_len: (i + 1) * part_len] for i in range(number_splits)]


def create_sequence_from_flattened_windows(X_raw, y_raw, window_size, overlap, sequence_length, feature_type='wavelet', wavelet='db4', level=4):
    """
    Create input/output sequences from raw sensor data for CNN/RNN/Transformer.

    Parameters:
    - X_raw: np.array (T, 8), piezo data
    - y_raw: np.array (T,), force data
    - window_size: number of samples in each sensor window
    - overlap: float in (0, 1), overlap ratio between windows
    - sequence_length: number of consecutive windows per sequence
    - feature_type: 'wavelet' or 'stft'
    
    Returns:
    - X_seq: np.array, shape (num_sequences, sequence_length, features_per_window)
    - y_seq: np.array, shape (num_sequences,)
    """
    step = int(window_size * (1 - overlap))

    def create_windows(data):
        return np.array([
            data[start:start + window_size]
            for start in range(0, len(data) - window_size + 1, step)
        ])

    X_windows = create_windows(X_raw)
    y_windows = create_windows(y_raw)
    y_windows = np.mean(y_windows, axis=1)

    def extract_wavelet_features(data):
        features = []
        for segment in data:
            coeffs = pywt.wavedec(segment, wavelet, level=level, axis=0)
            marginal = [np.sum(np.abs(c), axis=0) for c in coeffs[1:]]
            features.append(np.concatenate(marginal))
        return np.array(features)

    if feature_type == 'wavelet':
        feature_vectors = extract_wavelet_features(X_windows)
    else:
        raise NotImplementedError("Only wavelet implemented in this block.")

    # Create sequences of windows
    X_seq, y_seq = [], []
    for i in range(len(feature_vectors) - sequence_length):
        X_seq.append(feature_vectors[i:i + sequence_length])
        y_seq.append(y_windows[i + sequence_length - 1])

    return np.array(X_seq), np.array(y_seq)

# Feature extraction functions
def extract_wavelet_features(data, wavelet='db4', level=4):
    features = []
    for segment in data:
        coeffs = pywt.wavedec(segment, wavelet, level=level, axis=0)
        # Compute marginal coefficients (skip the approximation coefficients)
        marginal = [np.sum(np.abs(c), axis=0) for c in coeffs[1:]]
        features.append(np.concatenate(marginal))
    return np.array(features)

def extract_stft_features(data, fs=313, nperseg=64, noverlap=4):
    features = []
    for segment in data:
        f, t, Zxx = stft(segment, fs=fs, nperseg=nperseg, noverlap=noverlap)
        # Take the magnitude of the STFT and flatten it
        features.append(np.abs(Zxx).flatten())
    return np.array(features)

def create_windows(data, window_size, overlap):
    step = int(window_size * (1 - overlap))
    # step = 1
    windows = []
    for start in range(0, len(data) - window_size + 1, step):
        windows.append(data[start : start + window_size])
    return np.array(windows)

def build_model(num_layers, num_neurons, input_shape, model_type='simple'):
    if model_type == 'simple':
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        for _ in range(num_layers):
            model.add(Dense(num_neurons, activation='relu'))
        model.add(Dense(1))  # regression output

    elif model_type == 'cnn':
        model = Sequential()
        model.add(Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape))
        model.add(GlobalAveragePooling1D())
        for _ in range(num_layers):
            model.add(Dense(num_neurons, activation='relu'))
        model.add(Dense(1))

    elif model_type == 'rnn':
        model = Sequential()
        model.add(LSTM(num_neurons, return_sequences=False, input_shape=input_shape))
        for _ in range(num_layers - 1):
            model.add(Dense(num_neurons, activation='relu'))
        model.add(Dense(1))

    elif model_type == 'transformer':
        class TransformerBlock(tf.keras.layers.Layer):
            def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
                super().__init__()
                self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
                self.ffn = Sequential([
                    Dense(ff_dim, activation="relu"),
                    Dense(embed_dim)
                ])
                self.layernorm1 = LayerNormalization(epsilon=1e-6)
                self.layernorm2 = LayerNormalization(epsilon=1e-6)
                self.dropout1 = Dropout(rate)
                self.dropout2 = Dropout(rate)

            def call(self, inputs, training=None):
                attn_output = self.att(inputs, inputs)
                attn_output = self.dropout1(attn_output, training=training)
                out1 = self.layernorm1(inputs + attn_output)
                ffn_output = self.ffn(out1)
                ffn_output = self.dropout2(ffn_output, training=training)
                return self.layernorm2(out1 + ffn_output)

        input_layer = Input(shape=input_shape)
        x = TransformerBlock(embed_dim=input_shape[1], num_heads=2, ff_dim=num_neurons)(input_layer)
        x = GlobalAveragePooling1D()(x)
        for _ in range(num_layers):
            x = Dense(num_neurons, activation='relu')(x)
        output_layer = Dense(1)(x)
        model = Model(inputs=input_layer, outputs=output_layer)

    else:
        raise ValueError("Invalid model_type. Choose 'simple', 'cnn', 'rnn', or 'transformer'.")

    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])
    return model

# Create all 3-1-1 combinations (train, val, test)
all_indices = set(range(number_splits))
split_combinations = []
for train_combo in combinations(range(number_splits), number_splits - 2):  # choose 3 of 5
    remaining = list(all_indices - set(train_combo))
    val_idx, test_idx = remaining[0], remaining[1]
    split_combinations.append((list(train_combo), val_idx, test_idx))

best_score = float('inf')
best_params = {}
best_model = None

# Define the hyperparameter grid
param_grid = {
    'num_layers': [3],  # Number of hidden layers
    'num_neurons': [128]  # Number of neurons in each layer
}

# Perform manual grid search
best_score = float('inf')
best_params = {}

# Create windows for the training data

for i, (train_idxs, val_idx, test_idx) in enumerate(split_combinations):
    print(f"\n🔁 Split {i+1}: Train on {train_idxs}, Validate on {val_idx}, Test on {test_idx}")
    
    # Prepare raw data
    X_train_raw = np.vstack([X_parts[j] for j in train_idxs])
    y_train_raw = np.hstack([y_parts[j] for j in train_idxs])
    X_val_raw = X_parts[val_idx]
    y_val_raw = y_parts[val_idx]
    X_test_raw = X_parts[test_idx]
    y_test_raw = y_parts[test_idx]
    
    # Create windowed features
    if model_type == "simple":
        X_train_win = create_windows(X_train_raw, window_size, overlap)
        X_val_win = create_windows(X_val_raw, window_size, overlap)
        X_test_win = create_windows(X_test_raw, window_size, overlap)

        y_train_win = np.mean(create_windows(y_train_raw, window_size, overlap), axis=1)
        y_val_win = np.mean(create_windows(y_val_raw, window_size, overlap), axis=1)
        y_test_win = np.mean(create_windows(y_test_raw, window_size, overlap), axis=1)

        if feature_type == "wavelet":
            X_train_feat = extract_wavelet_features(X_train_win)
            X_val_feat = extract_wavelet_features(X_val_win)
            X_test_feat = extract_wavelet_features(X_test_win)
        elif feature_type == "stft":
            X_train_feat = extract_stft_features(X_train_win)
            X_val_feat = extract_stft_features(X_val_win)
            X_test_feat = extract_stft_features(X_test_win)
        else:
            raise NotImplementedError(...)

    else:
        X_train_feat, y_train_win = create_sequence_from_flattened_windows(X_train_raw, y_train_raw, window_size, overlap, sequence_length, feature_type)
        X_val_feat, y_val_win = create_sequence_from_flattened_windows(X_val_raw, y_val_raw, window_size, overlap, sequence_length, feature_type)
        X_test_feat, y_test_win = create_sequence_from_flattened_windows(X_test_raw, y_test_raw, window_size, overlap, sequence_length, feature_type)

    # Grid search over hyperparameters
    for num_layers in param_grid['num_layers']:
        for num_neurons in param_grid['num_neurons']:
            print(f"  ▶️ Training: layers={num_layers}, neurons={num_neurons}")
            input_shape_model = (X_train_feat.shape[1], X_train_feat.shape[2]) if model_type != "simple" else (X_train_feat.shape[1],)
            model = build_model(num_layers=num_layers, num_neurons=num_neurons, input_shape=input_shape_model, model_type=model_type)
            
            model.fit(X_train_feat, y_train_win, epochs=50, batch_size=32, validation_data=(X_val_feat, y_val_win), verbose=0)
            val_loss, val_mae = model.evaluate(X_val_feat, y_val_win, verbose=0)
            
            print(f"    ✅ Val MSE: {val_loss:.4f}, MAE: {val_mae:.4f}")

            if val_loss < best_score:
                best_score = val_loss
                best_params = {'num_layers': num_layers, 'num_neurons': num_neurons}
                best_model = model
                final_test_set = (X_test_feat, y_test_win)  # Store best test set


# === Final Evaluation ===
print(f"\n🏆 Best Hyperparameters: {best_params}")
print(f"📉 Best Validation Loss: {best_score}")

X_test_final, y_test_final = final_test_set
y_pred = best_model.predict(X_test_final)
y_pred_smoothed = np.convolve(y_pred.flatten(), np.ones(200)/200, mode='valid')

# === Plot Results ===
plt.figure(figsize=(10, 6))
plt.plot(y_pred_smoothed, label='Predicted Force')
plt.plot(y_test_final, label='True Force')
plt.xlabel('Sample Index')
plt.ylabel('Force')
plt.title(f'True vs Predicted Force ({feature_type}) - Best Model')
plt.legend()
plt.grid()
plt.show()