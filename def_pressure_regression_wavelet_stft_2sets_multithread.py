import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import pywt
from scipy.signal import stft
from joblib import Parallel, delayed
import multiprocessing
from keras.saving import register_keras_serializable

@register_keras_serializable(package='Custom', name='root_mean_squared_error')
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

#os.system('cls')
# Load the sensor data
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define parameters for the data folders
finger = "index"
test_n = 1
folder1 = f"{finger}_{test_n}"
folder1 = f"pressure_{test_n}"
folder2 = f"pressure_{test_n+1}"
test = "pressure"

piezo_data_path = os.path.join(script_dir, "data", test, folder1, f'{finger}.npy')
sensor_data_path = os.path.join(script_dir, "data", test, folder1, 'sensor_values_downsampled.npy')

piezo_data_path = os.path.join(script_dir, "data", "ar10", folder1, f'{finger}_downsampled.npy')
sensor_data_path = os.path.join(script_dir, "data", "ar10", folder1, 'sensor_values_downsampled.npy')

piezo_data_path2 = os.path.join(script_dir, "data", "ar10", folder2, f'{finger}_downsampled.npy')
sensor_data_path2 = os.path.join(script_dir, "data", "ar10", folder2, 'sensor_values_downsampled.npy')

sensor_data = np.load(sensor_data_path)
piezo_data = np.load(piezo_data_path)
sensor_data2 = np.load(sensor_data_path2)
piezo_data2 = np.load(piezo_data_path2)
sensor_data = sensor_data[:, 2]
sensor_data2 = sensor_data2[:, 2]

# Define the type of feature extraction
feature_type = "wavelet"  # Choose between "wavelet" or "stft"
print("sensor_data = ", sensor_data.shape)
print("piezo_data = ", piezo_data.shape)
print("sensor_data2 = ", sensor_data2.shape)
print("piezo_data2 = ", piezo_data2.shape)
# Preprocess the data
assert piezo_data.shape[0] == sensor_data.shape[0], "Mismatch in the number of samples 1"
assert piezo_data2.shape[0] == sensor_data2.shape[0], "Mismatch in the number of samples 2"

#test_size = 0.27  # 20% of the data will be used for testing
#split_index = int(len(piezo_data) * (1 - test_size))

# Split features (X) and labels (y)
X_train, X_test = piezo_data, piezo_data2
y_train, y_test = sensor_data, sensor_data2



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
    #print("first element of X_train = ", X_train[0])

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
    model.compile(optimizer='adam', loss=root_mean_squared_error)
    return model

seeds = np.random.choice(10000, size=66, replace=False).tolist()

# Define the hyperparameter grid
param_grid = {
    'num_layers': [3],  # Number of hidden layers
    'num_neurons': [128]  # Number of neurons in each layer
}

# Perform manual grid search
best_score = float('inf')
best_params = {}

num_cores = max(20, multiprocessing.cpu_count())

# Function to train and evaluate a model with given parameters
def train_and_evaluate_model(seed, num_layers, num_neurons):
    # Create a new TF graph and session for each parallel process
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    
    try:
        # Create and train the model
        model = create_model(num_layers=num_layers, num_neurons=num_neurons)
        history = model.fit(X_train, y_train_windows, epochs=50, batch_size=32, 
                          validation_split=0.2, verbose=0)

        # Evaluate the model on the test set
        test_loss = model.evaluate(X_test, y_test_windows, verbose=0)

        print(f"Test Loss (RMSE) with seed {seed}, {num_layers} layers, {num_neurons} neurons: {test_loss}")

        # Return only the score and parameters, not the model
        return (test_loss, seed, num_layers, num_neurons)
    except Exception as e:
        print(f"Error with seed {seed}: {str(e)}")
        return (float('inf'), seed, num_layers, num_neurons)

# Prepare tasks for parallel execution
tasks = [
    delayed(train_and_evaluate_model)(seed, num_layers, num_neurons)
    for seed in seeds
    for num_layers in param_grid['num_layers']
    for num_neurons in param_grid['num_neurons']
]

# Run the tasks in parallel
try:
    results = Parallel(n_jobs=num_cores)(tasks)
except Exception as e:
    print(f"Parallel execution failed: {str(e)}")
    # Handle the error or re-raise

# Find the best parameters based on the lowest test loss
best_score, best_seed, best_layers, best_neurons = min(results, key=lambda x: x[0])

# Now create the best model in the main process
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(best_seed)
best_model = create_model(num_layers=best_layers, num_neurons=best_neurons)
best_model.fit(X_train, y_train_windows, epochs=50, batch_size=32, 
              validation_split=0.2, verbose=1)

print(f"Best Parameters: Layers={best_layers}, Neurons={best_neurons}")
print(f"Best Test Loss (RMSE): {best_score}")
print(f"Best Seed: {best_seed}")

# Predict using the best model
print("X_test = ", X_test.shape)
y_pred = best_model.predict(X_test)
y_pred_original = np.convolve(y_pred.flatten(), np.ones(200)/200, mode='valid')

print("y_pred = ", y_pred.shape)
print(len(y_pred))
print(len(y_test_windows))
time_array = np.arange(0, len(y_pred_original)) * 1 / 313
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(time_array,-y_pred_original, label='Predicted Force', color='orange')
plt.plot(time_array,-y_test_windows, label='True Force', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.grid(True)
plt.title(f'True vs Predicted Force ({feature_type})')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(time_array,-y_pred_original, label='Predicted Force', color='orange')
plt.plot(time_array,-y_test_windows, label='True Force', color='blue')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.grid(False)
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