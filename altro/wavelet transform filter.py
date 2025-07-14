import numpy as np
import pywt
from scipy.signal import iirnotch, lfilter, butter
 
# Load the sensor data
sensor_values = np.load('C:/Users/utente/Documents/signals/pressure/index_1/index.npy')  # Finger sensors data

# Define the notch filter to remove 50Hz
def notch_filter(data, fs=1000, f0=50, Q=30):
    b, a = iirnotch(f0, Q, fs)
    filtered_data = lfilter(b, a, data)
    return filtered_data
 
# Define the low-pass filter
def low_pass_filter(data, cutoff=100, fs=1000, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data
 
# Apply the filters to each channel
filtered_data = []
for i in range(sensor_values.shape[1]):
    notch_filtered = notch_filter(sensor_values[:, i])
    low_pass_filtered = low_pass_filter(notch_filtered)
    filtered_data.append(low_pass_filtered)
 
filtered_data = np.array(filtered_data).T
 
# Apply Discrete Wavelet Transform
coeffs = []
for i in range(filtered_data.shape[1]):
    cA, cD = pywt.dwt(filtered_data[:, i], 'db1')
    coeffs.append((cA, cD))
 
# Prepare features for each channel
features = []
for cA, cD in coeffs:
    # Example feature extraction: mean and standard deviation
    features.append([np.mean(cA), np.std(cA), np.mean(cD), np.std(cD)])
 
features = np.array(features)