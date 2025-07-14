import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import pywt
from scipy.signal import stft

#os.system('cls')

# Load the sensor data
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define parameters for the data folders
finger = "thumb"
test_n = 2
folder1 = f"pressure_{finger}_{test_n}"
folder2 = f"pressure_{finger}_{test_n+1}"
test = "pressure"

piezo_data_path = os.path.join(script_dir, "data", test, folder1, f'{finger}_cleaned.npy')
sensor_data_path = os.path.join(script_dir, "data", test, folder1, 'sensor_values_downsampled.npy')

sensor_data = np.load(sensor_data_path)
piezo_data = np.load(piezo_data_path)
sensor_data = sensor_data[:, 2]

# Define the type of feature extraction
feature_type = "wavelet"  # Choose between "wavelet" or "stft"

# Preprocess the data
assert piezo_data.shape[0] == sensor_data.shape[0], "Mismatch in the number of samples"

test_size = 0.27  # 20% of the data will be used for testing
split_index = int(len(piezo_data) * (1 - test_size))

# Split features (X) and labels (y)
X_train, X_test = piezo_data[:split_index], piezo_data[split_index:]
y_train, y_test = sensor_data[:split_index], sensor_data[split_index:]



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
    step = 1
    windows = []
    for start in range(0, len(data) - window_size + 1, step):
        windows.append(data[start : start + window_size])
    return np.array(windows)

# Apply windowing and feature extraction to the training data
window_size = 400  # Example window size (adjust based on your data)
overlap = 0.9  # Example overlap (adjust based on your data)
print("X_train = ", X_train.shape)
# Create windows for the training data
X_train_windows = create_windows(X_train, window_size, overlap)
print("shape of X_train_windows = ", X_train_windows.shape)
X_test_windows = create_windows(X_test, window_size, overlap)
y_train_windows = create_windows(y_train, window_size, overlap)
y_test_windows = create_windows(y_test, window_size, overlap)
# Extract features based on the selected type
y_train_windows = np.mean(y_train_windows, axis=1)  # Or use y_train_windows[:, -1]
y_test_windows = np.mean(y_test_windows, axis=1)

if feature_type == "wavelet":
    X_train = extract_wavelet_features(X_train_windows)
    print("shape of X_train = ", X_train.shape)
    print("first element of X_train = ", X_train[0])
    exit()

    X_test = extract_wavelet_features(X_test_windows)
elif feature_type == "stft":
    X_train = extract_stft_features(X_train_windows)
    X_test = extract_stft_features(X_test_windows)
else:
    raise ValueError("Invalid feature type. Choose 'wavelet' or 'stft'.")

# Define the model creation function
def create_model(num_layers=1, num_neurons=32):
    model = Sequential()
    model.add(Dense(num_neurons, activation='relu', input_shape=(X_train.shape[1],)))  # Input layer
    
    # Add hidden layers
    for _ in range(num_layers - 1):
        model.add(Dense(num_neurons, activation='relu'))
    
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Define the hyperparameter grid
param_grid = {
    'num_layers': [3],  # Number of hidden layers
    'num_neurons': [128]  # Number of neurons in each layer
}

# Perform manual grid search
best_score = float('inf')
best_params = {}

for num_layers in param_grid['num_layers']:
    for num_neurons in param_grid['num_neurons']:
        print(f"Training model with {num_layers} layers and {num_neurons} neurons...")
        
        # Create and train the model
        model = create_model(num_layers=num_layers, num_neurons=num_neurons)
        history = model.fit(X_train, y_train_windows, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
        
        # Evaluate the model on the test set
        test_loss = model.evaluate(X_test, y_test_windows, verbose=0)
        print(f"Test Loss (MSE): {test_loss}")
        
        # Update the best model if this one is better
        if test_loss < best_score:
            best_score = test_loss
            best_params = {'num_layers': num_layers, 'num_neurons': num_neurons}
            best_model = model

# Print the best parameters and score
print(f"Best Parameters: {best_params}")
print(f"Best Test Loss (MSE): {best_score}")

# Predict using the best model
print("X_test = ", X_test.shape)
y_pred = best_model.predict(X_test)
y_pred_original = np.convolve(y_pred.flatten(), np.ones(200)/200, mode='valid')

print("y_pred = ", y_pred.shape)
print(len(y_pred))
print(len(y_test_windows))
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_pred_original, label='Predicted Force')
plt.plot(y_test_windows, label='True Force')
plt.xlabel('Sample Index')
plt.ylabel('Force')
plt.title(f'True vs Predicted Force ({feature_type})')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(y_pred_original[:23000], label='Predicted Force')
plt.plot(y_test_windows[:23000], label='True Force')
plt.xlabel('Sample Index')
plt.ylabel('Force')
plt.title(f'True vs Predicted Force ({feature_type})')
plt.legend()
plt.show()