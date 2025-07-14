import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import pywt
from scipy.signal import stft

# Load the sensor data
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define parameters for the data folders
finger = "thumb"
test_n = 1
folder1 = f"pressure_{finger}_{test_n}"
folder2 = f"pressure_{finger}_{test_n+1}"
test = "pressure"

piezo_data_path = os.path.join(script_dir, test, folder1, f'{finger}_cleaned.npy')
sensor_data_path = os.path.join(script_dir, test, folder1, 'sensor_values_downsampled.npy')

sensor_data = np.load(sensor_data_path)
piezo_data = np.load(piezo_data_path)
sensor_data = sensor_data[:, 2]

# Define the type of feature extraction
feature_type = "wavelet"  # Choose between "wavelet" or "stft"

# Preprocess the data
assert piezo_data.shape[0] == sensor_data.shape[0], "Mismatch in the number of samples"

test_size = 0.28  # 20% of the data will be used for testing
split_index = int(len(piezo_data) * (1 - test_size))

# Split features (X) and labels (y)
X_train, X_test = piezo_data[:split_index], piezo_data[split_index:]
y_train, y_test = sensor_data[:split_index], sensor_data[split_index:]

# Shuffle the training set (optional)
shuffle_indices = np.random.permutation(len(X_train))
X_train = X_train[shuffle_indices]
y_train = y_train[shuffle_indices]

# Feature extraction functions
def extract_wavelet_features(data, wavelet='db4', level=4):
    features = []
    print("data = ", (data).shape)
    for segment in data:
        coeffs = pywt.wavedec(segment, wavelet, level=level, axis=0)
        # Ensure all coefficients are at least 1D arrays
        marginal = [np.sum(np.abs(c), axis=0) for c in coeffs[1:]]
        
        features.append(np.concatenate(marginal))
    print("features = ", np.array(features).shape)
    return np.array(features)

def extract_stft_features(data, fs=313, nperseg=64, noverlap=32):
    features = []
    for segment in data:
        f, t, Zxx = stft(segment, fs=fs, nperseg=nperseg, noverlap=noverlap)
        # Take the magnitude of the STFT and flatten it
        features.append(np.abs(Zxx).flatten())
    return np.array(features)

# Windowing function
def create_windows(data, window_size, overlap):
    step = int(window_size * (1 - overlap))
    windows = []
    for start in range(0, len(data) - window_size + 1, step):
        windows.append(data[start : start + window_size])
    return np.array(windows)

# Apply windowing and feature extraction to the training data
window_size = 100  # Example window size (adjust based on your data)
overlap = 0.5  # Example overlap (adjust based on your data)
print("X_train = ", X_train.shape)
# Create windows for the training data
X_train_windows = create_windows(X_train, window_size, overlap)
X_test_windows = create_windows(X_test, window_size, overlap)
y_train_windows = create_windows(y_train, window_size, overlap)
y_test_windows = create_windows(y_test, window_size, overlap)

# Extract features for the training windows
if feature_type == "wavelet":
    X_train_features = extract_wavelet_features(X_train_windows)
    X_test_features = extract_wavelet_features(X_test_windows)
    print("X_train_features = ", X_train_features.shape)

elif feature_type == "stft":
    X_train_features = extract_stft_features(X_train_windows)
else:
    raise ValueError("Invalid feature type. Choose 'wavelet' or 'stft'.")


# Keep the test data as raw signal
X_test_raw = X_test
y_test_raw = y_test

# Define the model creation function
def create_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))  # Input layer
    
    # Add hidden layers
    model.add(Dense(64, activation='relu'))
    
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Create and train the model
model = create_model(input_shape=X_train_features.shape[1])
history = model.fit(X_train_features, y_train_windows, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate the model on the raw test data
# Since the test data is raw, we need to process it into windows and extract features
X_test_windows = create_windows(X_test_raw, window_size, overlap)
if feature_type == "wavelet":
    X_test_features = extract_wavelet_features(X_test_windows)
elif feature_type == "stft":
    X_test_features = extract_stft_features(X_test_windows)

# Flatten the labels for testing
y_test_features = create_windows(y_test_raw, window_size, overlap).mean(axis=1)

# Evaluate the model
test_loss = model.evaluate(X_test_features, y_test_features, verbose=0)
print(f"Test Loss (MSE): {test_loss}")

# Predict using the model
y_pred = model.predict(X_test_features)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_pred, label='Predicted Force')
plt.plot(y_test_features, label='True Force')
plt.xlabel('Sample Index')
plt.ylabel('Force')
plt.title('True vs Predicted Force')
plt.legend()
plt.show()