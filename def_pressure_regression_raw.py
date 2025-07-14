import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt



# Load the sensor data
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define parameters for the data folders
finger = "index"
test_n = 1
folder1 = f"{finger}_{test_n}"
folder2 = f"pressure_{test_n}"
test = "pressure"

piezo_data_path = os.path.join(script_dir, "data", test, folder1, f'{finger}.npy')
sensor_data_path = os.path.join(script_dir, "data", test, folder1, 'sensor_values_downsampled.npy')
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


# Shuffle the training set (optional)
# shuffle_indices = np.random.permutation(len(X_train))
# X_train = X_train[shuffle_indices]
# y_train = y_train[shuffle_indices]


# Define the model creation function
def create_model(num_layers=1, num_neurons=32):
    model = Sequential()
    model.add(Dense(num_neurons, activation='relu', input_shape=(8,)))  # Input layer
    
    # Add hidden layers
    for _ in range(num_layers - 1):
        model.add(Dense(num_neurons, activation='relu'))
    
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Define the hyperparameter grid
param_grid = {
    'num_layers': [3,2],  # Number of hidden layers
    'num_neurons': [128,64]  # Number of neurons in each layer
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

# # Inverse transform the normalized data
# y_pred_original = scaler_sensor.inverse_transform(y_pred)
# y_test_original = scaler_sensor.inverse_transform(y_test)

y_pred_original = np.convolve(y_pred.flatten(), np.ones(200)/200, mode='valid')

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_pred_original, label='Predicted Force')
plt.plot(y_test, label='True Force')
plt.xlabel('Sample Index')
plt.ylabel('Force')
plt.title('True vs Predicted Force (raw)')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(y_pred_original[:24000], label='Predicted Force')
plt.plot(y_test[:24000], label='True Force')
plt.xlabel('Sample Index')
plt.ylabel('Force')
plt.title('True vs Predicted Force (raw)')
plt.legend()
plt.show()