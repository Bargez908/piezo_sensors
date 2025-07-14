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

window_length = 0.2  # Window length in seconds
overlap = 0.5  # Overlap percentage (50%)
fs = 313  # Sampling frequency (Hz)
window_size = int(window_length * fs)  # Window size in samples
step_size = int(window_size * (1 - overlap))  # Step size in samples

# Preprocess the data
assert piezo_data.shape[0] == sensor_data.shape[0], "Mismatch in the number of samples"

test_size = 0.33  # 20% of the data will be used for testing
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
    for segment in data:
        print("segment = ", segment.shape)
        coeffs = pywt.wavedec(segment, wavelet, level=level)
        print("coeffs = ", len(coeffs))
        # Ensure all coefficients are at least 1D arrays
        coeffs = [np.atleast_1d(c) for c in coeffs]
        # Flatten and concatenate all detail coefficients (skip the approximation coefficients)
        marginal = [c.flatten() for c in coeffs[1:]]
        print("marginal = ", len(marginal))
        features.append(np.concatenate(marginal))
    return np.array(features)

def extract_stft_features(data, fs=313, nperseg=64, noverlap=32):
    features = []
    for segment in data:
        f, t, Zxx = stft(segment, fs=fs, nperseg=nperseg, noverlap=noverlap)
        # Take the magnitude of the STFT and flatten it
        features.append(np.abs(Zxx).flatten())
    return np.array(features)

# Extract features based on the selected type
if feature_type == "wavelet":
    X_train = extract_wavelet_features(X_train.T)
    exit()
    X_test = extract_wavelet_features(X_test)
elif feature_type == "stft":
    X_train = extract_stft_features(X_train)
    X_test = extract_stft_features(X_test)
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
    'num_layers': [2],  # Number of hidden layers
    'num_neurons': [64]  # Number of neurons in each layer
}

# Perform manual grid search
best_score = float('inf')
best_params = {}

for num_layers in param_grid['num_layers']:
    for num_neurons in param_grid['num_neurons']:
        print(f"Training model with {num_layers} layers and {num_neurons} neurons...")
        
        # Create and train the model
        model = create_model(num_layers=num_layers, num_neurons=num_neurons)
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
        
        # Evaluate the model on the test set
        test_loss = model.evaluate(X_test, y_test, verbose=0)
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
y_pred = best_model.predict(X_test)

y_pred_original = np.convolve(y_pred.flatten(), np.ones(200)/200, mode='valid')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_pred_original, label='Predicted Force')
plt.plot(y_test, label='True Force')
plt.xlabel('Sample Index')
plt.ylabel('Force')
plt.title('True vs Predicted Force')
plt.legend()
plt.show()