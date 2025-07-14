import numpy as np
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
import os

# Load the sensor data
script_dir = os.path.dirname(os.path.abspath(__file__))
finger1="index_1"

# Define the relative path to the file
sensor_data_path = os.path.join(script_dir, 'pressure', finger1, 'index.npy')
idle_points_path = os.path.join(script_dir, 'pressure', finger1, 'labels', 'idle_points.npy')
rise_points_path = os.path.join(script_dir, 'pressure', finger1, 'labels', 'rise_points.npy')
steady_points_path = os.path.join(script_dir, 'pressure', finger1, 'labels', 'steady_points.npy')
fall_points_path = os.path.join(script_dir, 'pressure', finger1, 'labels', 'fall_points.npy')

# Load the sensor data and labels using the relative paths
sensor_values_filtered = np.load(sensor_data_path)
idle_points = np.load(idle_points_path)
rise_points = np.load(rise_points_path)
steady_points = np.load(steady_points_path)
fall_points = np.load(fall_points_path)


# Map points to labels
signal_length = len(sensor_values_filtered)


def segment_signal(signal, fs, window_length, overlap):
    step = int(window_length * fs * (1 - overlap))  # Ensure step is an integer
    window_size = int(window_length * fs)  # Ensure window size is an integer
    segments = []
    for start in range(0, len(signal) - window_size + 1, step):
        segments.append(signal[start : start + window_size])
    return np.array(segments)



window_lengths = [0.1, 0.2, 0.5, 1, 2]  # in seconds
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
    [idle_points, rise_points, steady_points, fall_points], [0, 1, 1, 1]
):
    label_steps[label_array] = label_value
plt.plot(label_steps)
plt.xlabel("Samples")
plt.ylabel("Class")
plt.title("Class Labels")
plt.grid()
plt.show()

 # Assign labels to segments
def assign_labels_to_segments(segments, fs, window_length, overlap, labels):
    step = int(window_length * fs * (1 - overlap))
    window_size = int(window_length * fs)
    segment_labels = []
    
    for start in range(0, len(labels) - window_size + 1, step):
        # Extract the window of labels
        label_window = labels[start : start + window_size]
        # Determine the most frequent label
        most_common_label = mode(label_window)[0]
        segment_labels.append(most_common_label)
    
    return np.array(segment_labels)


labeled_segments = {}
for wl in window_lengths:
    segment_labels = assign_labels_to_segments(
        segments[wl], fs, wl, overlap, label_steps
    )
    labeled_segments[wl] = segment_labels

print("Segment labels for 2s window:", len(labeled_segments[window_lengths[0]]))
print("Segment labels for 4s window:", len(labeled_segments[window_lengths[1]]))
print("Segment labels for 8s window:", len(labeled_segments[window_lengths[2]]))

import pywt

# DOING MARGINAL AND CREATUBG FEATURES 
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

# Combine features and labels
labeled_features = {}

for wl in window_lengths:
    labeled_features[wl] = {}
    for w in wavelets:
        # Extract features for the current window length and wavelet
        feature_matrix = features[wl][w]
        
        # Ensure that the number of segments matches the labels
        if len(labeled_segments[wl]) != feature_matrix.shape[0]:
            raise ValueError(
                f"Mismatch between features and labels for window length {wl}s and wavelet {w}: "
                f"{feature_matrix.shape[0]} features vs {len(labeled_segments[wl])} labels."
            )
        
        # Append labels as the last column
        labeled_feature_matrix = np.hstack((feature_matrix, labeled_segments[wl][:, None]))
        labeled_features[wl][w] = labeled_feature_matrix

# Example: Print the shape of the labeled features
for wl in window_lengths:
    for w in wavelets:
        print(f"Labeled Feature Matrix - Window Length: {wl}s, Wavelet: {w}, Shape: {labeled_features[wl][w].shape}")

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score

# Initialize results storage
results = []

# Loop through all combinations of window lengths and wavelets
for wl in window_lengths:
    for w in wavelets:
        # Get the labeled feature matrix for the current combination
        labeled_matrix = labeled_features[wl][w]
        
        # Separate features and labels
        X = labeled_matrix[:, :-1]  # Features (all columns except the last)
        y = labeled_matrix[:, -1]   # Labels (last column)
        
        # Normalize the features
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
        
        # Initialize the SVM classifier
        clf = SVC(kernel='rbf', gamma='scale', C=1.0)
        
        # Perform Stratified K-Fold Cross-Validation
        skf = StratifiedKFold(n_splits=10)
        all_precisions = []
        
        for train_index, test_index in skf.split(X_normalized, y):
            X_train, X_test = X_normalized[train_index], X_normalized[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Train the model
            clf.fit(X_train, y_train)
            
            # Predict on the test set
            y_pred = clf.predict(X_test)
            
            # Compute precision
            precision = precision_score(y_test, y_pred, average='weighted')  # Use 'weighted' for multi-class
            all_precisions.append(precision)
        
        # Compute mean and standard deviation of precision
        mean_precision = np.mean(all_precisions)
        std_precision = np.std(all_precisions)
        
        # Store results
        results.append({
            "window_length": wl,
            "wavelet": w,
            "mean_precision": mean_precision,
            "std_precision": std_precision,
            "model": clf,
            "scaler": scaler,
            "X": X_normalized,
            "y": y
        })

# Find the best combination
best_result = max(results, key=lambda x: x["mean_precision"])

# Print all results and the best combination
for result in results:
    print(
        f"Window Length: {result['window_length']}s, Wavelet: {result['wavelet']}, "
        f"Precision: {result['mean_precision']:.4f} ± {result['std_precision']:.4f}"
    )

print("\nBest Combination:")
print(
    f"Window Length: {best_result['window_length']}s, Wavelet: {best_result['wavelet']}, "
    f"Precision: {best_result['mean_precision']:.4f} ± {best_result['std_precision']:.4f}"
)

# Confusion Matrix for the Best Combination
clf_best = best_result["model"]
scaler_best = best_result["scaler"]
X_best = best_result["X"]
y_best = best_result["y"]

# Perform train-test split for final evaluation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_best, y_best, test_size=0.2, stratify=y_best, random_state=42)

# Train on the training set
clf_best.fit(X_train, y_train)

# Predict on the test set
y_pred_best = clf_best.predict(X_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
labels = np.unique(y_best)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title(f"Confusion Matrix (Best Combination: {best_result['window_length']}s, {best_result['wavelet']})")
plt.show()