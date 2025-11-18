
import numpy as np
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
from scipy.stats import mode
from scipy.signal import butter, filtfilt
import os
from sklearn.svm import SVC
from hmmlearn import hmm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import joblib
import time
from scipy.signal import stft
from itertools import product

def equalize_labels(label_points, piezo_values):
    # Step 1: Find the label with the fewest values
    unique_labels, label_counts = np.unique(label_points, return_counts=True)
    min_count = min(label_counts)
    print("Unique labels:", unique_labels)
    print("Label counts:", label_counts)
    print("Minimum count:", min_count)
    
    # Step 2: For each label, pop values until the number of occurrences matches the minimum count
    labels_to_remove = {label: count - min_count for label, count in zip(unique_labels, label_counts) if count > min_count}
    
    # Step 3: Iterate through the labels and remove from both label_points and piezo_values
    label_indices_to_remove = []
    for label, count_to_remove in labels_to_remove.items():
        indices = np.where(label_points == label)[0]
        for i in range(count_to_remove):
            label_indices_to_remove.append(indices[-(i+1)])

    # Step 4: Sort indices in reverse order to avoid shifting when popping
    label_indices_to_remove.sort(reverse=True)

    # Step 5: Pop values from both label_points and piezo_values

    for index in label_indices_to_remove:
        label_points = np.delete(label_points, index)
        piezo_values = np.delete(piezo_values, index, axis=0)
    
    return label_points, piezo_values

script_dir = os.path.dirname(os.path.abspath(__file__))
# Define parameters for the data folders
# === Configuration ===
window_size = 300              # Number of samples per window
overlap = True                 # Enable or disable overlap
overlap_value = 0.8            # % overlap (0.0 to 1.0) if overlap is True
finger = "thumb"
test_n = 0
test = f"{finger}_level_2"
folder1 = f'sliding_{test_n}'
unify_labels = True  # Set to True to unify labels
unified_labels = {0: [0], 1: [1,2,3,4], 2: [5,6]}  # 1,2,3,4 are sliding, 5,6 are rotation

# Define file paths for the first dataset
piezo_data_path = os.path.join(script_dir,"data\\records_final", test, "sliding", folder1, f'{finger}_cleaned.npy')
labels_data_path = os.path.join(script_dir,"data\\records_final", test, "sliding", folder1, 'labels.npy')

piezo_values = np.load(piezo_data_path)

label_points = np.load(labels_data_path)

if unify_labels:
    remapped_labels = label_points.copy()
    for new_label, old_labels in unified_labels.items():
        for old_label in old_labels:
            remapped_labels[label_points == old_label] = new_label
    label_points = remapped_labels

print("\nlen label_points", len(label_points))
label_points, piezo_values = equalize_labels(label_points, piezo_values)
print("len label_points after equalizing", len(label_points),"\n")

# Create a base output directory for this test
output_base_dir = os.path.join(script_dir, "csv_nas", test)
os.makedirs(output_base_dir, exist_ok=True)

# Loop through each label and save corresponding piezo_values to CSV
unique_labels = np.unique(label_points)

for label in unique_labels:
    label_indices = np.where(label_points == label)[0]
    label_piezo_data = piezo_values[label_indices].astype(int)  # Convert to integers

    # Compute step size based on overlap
    step = int(window_size * (1 - overlap_value)) if overlap else window_size

    # Compute number of full windows
    total_samples = label_piezo_data.shape[0]
    num_windows = (total_samples - window_size) // step + 1

    print(f"🔍 Label {label}: {total_samples} samples → {num_windows} windows")

    # Create folder for this label
    label_dir = os.path.join(output_base_dir, f"{label}")
    os.makedirs(label_dir, exist_ok=True)

    # Header row
    num_columns = piezo_values.shape[1]
    header = ",".join(str(i) for i in range(num_columns))

    # Create and save each window
    for n in range(num_windows):
        start = n * step
        end = start + window_size
        window = label_piezo_data[start:end]

        if window.shape[0] != window_size:
            continue  # skip incomplete windows

        csv_path = os.path.join(label_dir, f"{test.replace('\\', '_')}_{n}.csv")
        np.savetxt(csv_path, window, delimiter=",", fmt="%d", header=header, comments='')

    print(f"✅ Saved {num_windows} windows for label {label} in: {label_dir}")