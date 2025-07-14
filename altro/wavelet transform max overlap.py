import numpy as np
import pywt
import matplotlib.pyplot as plt
import seaborn as sns

# Load the sensor data
sensor_values_filtered = np.load('C:/Users/bargi/Downloads/signals/pressure/index_1/index.npy')  # Finger sensors data

idle_points = np.load('C:/Users/bargi/Downloads/signals/pressure/index_1/labels/idle_points.npy')
rise_points = np.load('C:/Users/bargi/Downloads/signals/pressure/index_1/labels/rise_points.npy')
steady_points = np.load('C:/Users/bargi/Downloads/signals/pressure/index_1/labels/steady_points.npy')
fall_points = np.load('C:/Users/bargi/Downloads/signals/pressure/index_1/labels/fall_points.npy')

# Map points to labels
signal_length = len(sensor_values_filtered)
labels_per_step = np.zeros(signal_length, dtype=int)  # Default to class 0 (idle)
labels_per_step[rise_points] = 1
labels_per_step[steady_points] = 1
labels_per_step[fall_points] = 1


def segment_signal(signal, fs, window_length, overlap):
    step = int(window_length * fs * (1 - overlap))  # Ensure step is an integer
    window_size = int(window_length * fs)  # Ensure window size is an integer
    segments = []
    for start in range(0, len(signal) - window_size + 1, step):
        segments.append(signal[start : start + window_size])
    return np.array(segments)



window_lengths = [2, 4, 8]  # in seconds
fs=313
overlap = 0.5
segments = {wl: segment_signal(sensor_values_filtered, fs, wl, overlap) for wl in window_lengths}

# Plot example segments for a specific window length
# example_segments = segments[0.2]  # 200 ms window
# plt.figure(figsize=(12, 6))
# for i in range(min(5, len(example_segments))):  # Plot first 5 segments
#     plt.plot(example_segments[i][:, 0], label=f"Segment {i+1}")
# plt.xlabel("Samples")
# plt.ylabel("Amplitude")
# plt.title("Example Segments (200 ms, First Channel)")
# plt.legend()
# plt.grid()
# plt.show()

label_steps = np.zeros(len(sensor_values_filtered))
for label_array, label_value in zip(
    [idle_points, rise_points, steady_points, fall_points], [0, 1, 2, 3]
):
    label_steps[label_array] = label_value
plt.plot(label_steps)
plt.xlabel("Samples")
plt.ylabel("Class")
plt.title("Class Labels")
plt.grid()
plt.show()

 # Assign labels to segments
def assign_labels_to_segments(segments, fs, window_length, overlap, labels_per_step):
    step = int(window_length * fs * (1 - overlap))
    window_size = int(window_length * fs)
    segment_labels = []
    
    for start in range(0, len(labels_per_step) - window_size + 1, step):
        segment_steps = labels_per_step[start : start + window_size]
        label = np.bincount(segment_steps).argmax()  # Majority class
        segment_labels.append(label)
    
    return np.array(segment_labels)

segment_labels = {
    wl: assign_labels_to_segments(segments[wl], fs, wl, overlap, labels_per_step)
    for wl in window_lengths
}


import pywt

def extract_wavelet_features(segments, wavelet, level=4):
    features = []
    for segment in segments:
        coeffs = pywt.wavedec(segment, wavelet, level=level, axis=0)
        # Compute marginal coefficients for each decomposition level
        marginal = [np.sum(np.abs(c), axis=0) for c in coeffs[1:]]  # Skip approximation coeffs
        features.append(np.concatenate(marginal))
    return np.array(features)

wavelets = ['db2', 'db3', 'db4', 'sym2', 'sym3', 'sym4', 'coif2', 'coif3', 'coif4']
features = {
    wl: {w: extract_wavelet_features(segments[wl], w) for w in wavelets}
    for wl in window_lengths
}
#i want to print the shape of all feature 
for wl in window_lengths:
    for w in wavelets:
        print(f"Window length: {wl}s, Wavelet: {w}, Shape: {features[wl][w].shape}")
exit()

def create_feature_matrix(features, labels=None):
    feature_matrix = []
    for feature_set in features:
        if labels is not None:
            feature_matrix.append(np.hstack((feature_set, labels)))
        else:
            feature_matrix.append(feature_set)
    return np.vstack(feature_matrix)


# Assuming labels are available
# Generate labels (dummy example, replace with actual labels)
labels = np.random.randint(0, 7, size=len(features))
feature_matrix = create_feature_matrix(features[2]['db3'], labels)

# Select the features for the desired window length and wavelet
selected_features = features[2]['db3']  # Example for window length 2 and wavelet 'db3'

# Convert features to a NumPy array and slice the first 10 rows
heatmap_data = selected_features[:10]

# Create the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap="viridis", annot=False)
plt.xlabel("Features")
plt.ylabel("Segments")
plt.title("Heatmap of Features for the First 10 Segments")
plt.show()


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(feature_matrix[:, :-1])
y = feature_matrix[:, -1]

clf = SVC(kernel='rbf', gamma='scale', C=1.0)
scores = cross_val_score(clf, X, y, cv=10)

print("Cross-validation accuracy:", np.mean(scores))
