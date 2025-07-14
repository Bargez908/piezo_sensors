import numpy as np
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
import os

# Creazione di un semplice grafico
script_dir = os.path.dirname(os.path.abspath(__file__))

finger="index"
test_n = 1
folder1=f"rolling_{test_n}_{finger}"
folder2=f"rolling_{test_n+1}_{finger}"
test = "rolling"

# Define the relative path to the file
piezo_data_path = os.path.join(script_dir, test, folder1, f'{finger}_cleaned.npy')
trigger_data_path = os.path.join(script_dir, test, folder1, 'triggers_cleaned.npy')
labels_data_path = os.path.join(script_dir, test, folder1, 'labels.npy')

piezo_data_path_2 = os.path.join(script_dir, test, folder2, f'{finger}_cleaned.npy')
trigger_data_path_2 = os.path.join(script_dir, test, folder2, 'triggers_cleaned.npy')
labels_data_path_2 = os.path.join(script_dir, test, folder2, 'labels.npy')
# Load the sensor data and labels using the relative paths

piezo_values = np.load(piezo_data_path)
label_points = np.load(labels_data_path)

piezo_values2 = np.load(piezo_data_path_2)
label_points2 = np.load(labels_data_path_2)

# Map points to labels
signal_length = len(piezo_values)
signal_length2 = len(piezo_values2)
print(label_points.shape)
print(label_points2.shape)
print(piezo_values.shape)
 
def segment_signal(signal, fs, window_length, overlap):
    step = int(window_length * fs * (1 - overlap))  # Ensure step is an integer
    #step=1
    window_size = int(window_length * fs)  # Ensure window size is an integer
    segments = []
    for start in range(0, len(signal) - window_size + 1, step):
        segments.append(signal[start : start + window_size])
    return np.array(segments)



window_lengths = [1, 2, 2.3,  2.5,]  # in seconds
fs=313
overlaps = [0.6, 0.65, 0.7]  # Example overlap values
segments = {wl: {ov: segment_signal(piezo_values, fs, wl, ov) for ov in overlaps} for wl in window_lengths}
segments2 = {wl: {ov: segment_signal(piezo_values2, fs, wl, ov) for ov in overlaps} for wl in window_lengths}


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

#plot both label steps
plt.plot(label_points)
plt.plot(label_points2)
plt.xlabel("Samples")
plt.ylabel("Class")
plt.title("Class Labels")
plt.grid()
plt.show()
 # Assign labels to segments
def assign_labels_to_segments(segments, fs, window_length, overlap, labels):
    step = int(window_length * fs * (1 - overlap))
    #step=1
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
labeled_segments2 = {}
for wl in window_lengths:
    labeled_segments[wl] = {}
    labeled_segments2[wl] = {}
    for ov in overlaps:
        segment_labels = assign_labels_to_segments(
            segments[wl][ov], fs, wl, ov, label_points
        )
        labeled_segments[wl][ov] = segment_labels
        segment_labels2 = assign_labels_to_segments(
            segments2[wl][ov], fs, wl, ov, label_points2
        )
        labeled_segments2[wl][ov] = segment_labels2 


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

wavelets = ['db4','sym3', 'sym4', 'sym5', 'coif5']
features = {
    wl: {ov: {w: extract_wavelet_features(segments[wl][ov], w) for w in wavelets} for ov in overlaps}
    for wl in window_lengths
}
features2 = {
    wl: {ov: {w: extract_wavelet_features(segments2[wl][ov], w) for w in wavelets} for ov in overlaps}
    for wl in window_lengths
}

#i want to print the shape of all feature 
for wl in window_lengths:
    for ov in overlaps:  # Iterate over overlaps
        for w in wavelets:
            print(f"Window length: {wl}s, Overlap: {ov}, Wavelet: {w}, Shape: {features[wl][ov][w].shape}")

# Combine features and labels
labeled_features = {}
labeled_features2 = {}

for wl in window_lengths:
    labeled_features[wl] = {}
    labeled_features2[wl] = {}
    for ov in overlaps:  # Iterate over overlaps
        for w in wavelets:
            # Extract features for the current window length, overlap, and wavelet
            feature_matrix = features[wl][ov][w]
            feature_matrix2 = features2[wl][ov][w]

            # Ensure that the number of segments matches the labels
            if len(labeled_segments[wl][ov]) != feature_matrix.shape[0]:
                raise ValueError(
                    f"Mismatch between features and labels for window length {wl}s, overlap {ov}, and wavelet {w}: "
                    f"{feature_matrix.shape[0]} features vs {len(labeled_segments[wl][ov])} labels."
                )

            # Append labels as the last column
            print("Feature matrix shape:", feature_matrix.shape)
            print("Labeled segment shape before reshape:", labeled_segments[wl][ov].shape)

            labeled_feature_matrix = np.hstack((feature_matrix, labeled_segments[wl][ov][:, None]))
            labeled_feature_matrix2 = np.hstack((feature_matrix2, labeled_segments2[wl][ov][:, None]))
            labeled_features[wl].setdefault(ov, {})[w] = labeled_feature_matrix
            labeled_features2[wl].setdefault(ov, {})[w] = labeled_feature_matrix2


# Example: Print the shape of the labeled features
for wl in window_lengths:
    for ov in overlaps:  # Iterate over overlaps
        for w in wavelets:
            print(f"Labeled Feature Matrix - Window Length: {wl}s, Overlap: {ov}, Wavelet: {w}, Shape: {feature_matrix.shape}")

from sklearn.svm import SVC
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score

# Initialize results storage
results = []
results_test_on_set2 = []

C_values = [0.1, 1, 10, 50]  # Example range for the regularization parameter
gamma_values = ['scale', 0.01, 0.1]  # Example range for the kernel coefficient
kernels = ['linear', 'poly']  # Different kernel types

# Loop through all combinations of window lengths and wavelets
with tqdm(total=len(window_lengths) * len(wavelets) * len(overlaps) * len(C_values) * len(gamma_values) * len(kernels), desc="Processing", unit="comb") as pbar:    
    for wl in window_lengths:
        for w in wavelets:
            for ov in overlaps:
                # Get the labeled feature matrices for the current combination
                labeled_matrix = labeled_features[wl][ov][w]
                labeled_matrix2 = labeled_features2[wl][ov][w]

                # Separate features and labels
                X = labeled_matrix[:, :-1]  # Features (all columns except the last)
                y = labeled_matrix[:, -1]   # Labels (last column)
                X2 = labeled_matrix2[:, :-1]  # Features (all columns except the last)
                y2 = labeled_matrix2[:, -1]   # Labels (last column)

                # Normalize the features (you can skip normalization here if unnecessary)
                X_train = X
                X_test = X2

                for C in C_values:
                    for gamma in gamma_values:
                        for kernel in kernels:
                            # Train the SVM classifier with the current parameters
                            clf = SVC(kernel=kernel, gamma=gamma, C=C)
                            clf.fit(X_train, y)

                            # Predict on the test set (X_test)
                            y_pred = clf.predict(X_test)

                            # Compute precision
                            precision = precision_score(y2, y_pred, average='weighted')  # Use 'weighted' for multi-class

                            # Generate the confusion matrix
                            cm = confusion_matrix(y2, y_pred)

                            # Store the results
                            results_test_on_set2.append({
                                "window_length": wl,
                                "wavelet": w,
                                "overlap": ov,
                                "C": C,
                                "gamma": gamma,
                                "kernel": kernel,
                                "precision": precision,
                                "confusion_matrix": cm
                            })
                            pbar.update(1)

# Find the best combination based on precision
best_result_test_on_set2 = max(results_test_on_set2, key=lambda x: x["precision"])

sorted_results = sorted(results_test_on_set2, key=lambda x: x["precision"], reverse=True)

# Print results for all combinations
for result in results_test_on_set2:
    print(
        f"Window Length: {result['window_length']}s, Wavelet: {result['wavelet']}, Overlap: {result['overlap']}, "
        f"Kernel: {result['kernel']}, Gamma: {result['gamma']}, C: {result['C']}, Precision: {result['precision']:.4f}"
    )

# Print the top 10 combinations in reverse order (from 10 to 1)
print("\nTop 30 Combinations (Test on Set 2):")
for i, result in enumerate(reversed(sorted_results[:30]), start=1):
    print(
        f"Rank {30 - i + 1}: Window Length: {result['window_length']}s, Wavelet: {result['wavelet']}, "
        f"Overlap: {result['overlap']}, Kernel: {result['kernel']}, Gamma: {result['gamma']}, C: {result['C']}, "
        f"Precision: {result['precision']:.4f}"
    )

# Plot the confusion matrix for the best combination
best_cm = best_result_test_on_set2["confusion_matrix"]
plt.figure(figsize=(8, 6))
sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y2), yticklabels=np.unique(y2))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

# Include the SVM parameters in the title
plt.title(
    f"Confusion Matrix (Best Combination: {best_result_test_on_set2['window_length']}s, "
    f"Wavelet: {best_result_test_on_set2['wavelet']}, Overlap: {best_result_test_on_set2['overlap']}, "
    f"Kernel: {best_result_test_on_set2['kernel']}, Gamma: {best_result_test_on_set2['gamma']}, C: {best_result_test_on_set2['C']})"
)

plt.show()
