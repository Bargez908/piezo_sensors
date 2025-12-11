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
from sklearn.model_selection import train_test_split

def assign_labels_to_segments(segments, fs, window_length, overlap, labels):
    """Assign a label to each segment only if all labels in the window are the same. Otherwise, discard the window."""
    if overlap == 1.0:
        step = 1
    else:
        step = int(window_length * fs * (1 - overlap))  # step size (samples)
    window_size = int(window_length * fs)
    segment_labels = []
    valid_segments = []
    
    # Count the total number of segments before filtering
    total_segments = len(range(0, len(labels) - window_size + 1, step))
    
    for start in range(0, len(labels) - window_size + 1, step):
        label_window = labels[start : start + window_size]
        # Check if all labels in the window are the same
        if np.all(label_window == label_window[0]):
            segment_labels.append(label_window[0])
            valid_segments.append(segments[start // step])  # Keep the corresponding segment
        else:
            # Discard the window if labels are not consistent
            continue
    
    # Print the number of segments before and after filtering
    print(
        f"Window Length: {window_length}s, Overlap: {overlap} - "
        f"Total Segments: {total_segments}, Valid Segments: {len(valid_segments)}"
    )
    
    return np.array(valid_segments), np.array(segment_labels)


# if os.name == 'nt':
#     os.system('cls')

# ---------------------------
# 1. Data Loading and Setup
# ---------------------------
# Define the type of feature extraction
feature_type = "ar"  # Choose between "wavelet" or "stft" or "integral" or "ar" or "nothing"
treshold = False  # Apply a treshold to the piezo values
filter_type = "nothing"  # Choose between "highpass" or "lowpass" or "ema" or "nothing"
freq = 20  # Frequency for the highpass or lowpass filter
equalize_labels_toggle = True  # Equalize the number of samples for each class
remap_labels_toggle = False  # Remap labels into fewer classes, look at the remapping function to change the mapping
pop_values_toggle = False  # Pop values of the chosen labels
labels_to_pop = [1,2,3,4]  # Labels to pop from the dataset


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define parameters for the data folders
finger = "thumb"
test_n = 1
test = "pressure"
folder1 = f"{test}_{finger}_{test_n}"
folder2 = f"{test}_{finger}_{test_n+1}"
classifier = "svm" # Choose between "svm" or "hmm"

AR_orders = [5, 10, 15]
AR_window_lengths = [25, 100, 400]
AR_overlaps = [50, 100]   # percentuali, NON 0.5

# Define segmentation parameters for non ar features
window_lengths = [0.2, 0.4]  # in seconds
fs = 313
overlaps = [0.4, 0.6]  # Overlap percentage

# Define the wavelet families to use (only used if feature_type == "wavelet")
wavelets = ['db2', 'db3', 'db4', 'sym2', 'sym3', 'sym4', 'coif2', 'coif3', 'coif4']

# Define SVM hyperparameters
C_values = [2, 1, 1/2]
gamma_values = ['scale']
kernels = ['rbf']
# initial_transition_matrix = np.array([[0.9, 0.1, 0, 0], 
#                                       [0, 0.9, 0.1, 0],
#                                       [0, 0, 0.9, 0.1],
#                                       [0.1, 0, 0, 0.9]])  # Custom transition matrix for unsupervised HMM training
# initial_start_prob = np.array([1, 0, 0, 0])  # Initial start probabilities for unsupervised HMM training
initial_transition_matrix = np.array([[0.99, 0.01],
                                      [0, 1]])  # Custom transition matrix for unsupervised HMM training
initial_start_prob = np.array([1, 0])  # Initial start probabilities for unsupervised HMM training
unsupervised = True  # If True, use unsupervised training for HMM
n_states_values = [3]  # used for supervised training

# Define file paths for the first dataset
piezo_data_path = os.path.join(script_dir,"data", test, folder1, f'{finger}_cleaned.npy')
labels_data_path = os.path.join(script_dir,"data", test, folder1, 'labels2.npy')

# Define file paths for the second dataset
piezo_data_path_2 = os.path.join(script_dir,"data", test, folder2, f'{finger}_cleaned.npy')
labels_data_path_2 = os.path.join(script_dir,"data", test, folder2, 'labels2.npy')

if classifier == "svm":
    best_model_path = os.path.join(script_dir, test, f"best_svm_model_{type}.joblib")
elif classifier == "hmm":
    best_model_path = os.path.join(script_dir, test, f"best_hmm_model_{type}.joblib")


def load_ar_set(base_path, order, wl, ov):
    """
    Load AR coefficients and noise variances for a given (order, window_len, overlap).
    Returns (AR_full_features, labels)
    """
    name = f"{order}_{wl}_{ov}"

    ar_path = os.path.join(base_path, "ar_coeff", f"{name}_arCoeffs.npy")
    noise_path = os.path.join(base_path, "ar_coeff", f"{name}_noiseVariances.npy")
    labels_path = os.path.join(base_path, "labels.npy")  # you must save labels for each
    AR_coeffs = np.load(ar_path)
    noiseVar = np.load(noise_path)
    labels = np.load(labels_path)

    # Remove AR(0) coeff = 1
    # print("AR_coeffs shape:", AR_coeffs.shape)
    # print("noiseVar shape:", noiseVar.shape)
    AR_trim = AR_coeffs[:, 1:, :]
    noise_exp = noiseVar[:, None, :]
    # print("AR_trim shape:", AR_trim.shape)
    # print("noise_exp shape:", noise_exp.shape)
    AR_full = np.concatenate([AR_trim, noise_exp], axis=1)

    N = AR_full.shape[2]
    features = AR_full.transpose(2,0,1).reshape(N, -1)

    return features, labels

def window_labels_for_ar(labels_full, window_length, overlap):
    """
    Ricrea il windowing delle labels usando la stessa logica
    con cui è stato creato l'AR.
    Restituisce:
       valid_features_idx  -> indici AR da mantenere
       window_labels       -> label per ogni finestra valida
    """
    stepSize = max(1, int(window_length * (1 - overlap)))
    segments = []
    start = 0

    while start + window_length <= len(labels_full):
        segments.append(np.arange(start, start + window_length))
        start += stepSize

    segments = np.array(segments)  # shape = [N_windows × window_length]

    window_labels = []
    for seg in segments:
        seg_lab = labels_full[seg]
        values = np.unique(seg_lab)

        # finestra valida solo se pura
        if len(values) == 1:
            window_labels.append(values[0])
        else:
            window_labels.append(-1)

    window_labels = np.array(window_labels)
    valid_idx = np.where(window_labels != -1)[0]

    return valid_idx, window_labels

def pop_values(labels, features, labels_to_pop):
    """
    labels:  1D array (n_windows,)
    features: 2D array (n_windows, n_features)
    labels_to_pop: lista/array di classi da eliminare
    """
    mask = ~np.isin(labels, labels_to_pop)
    return labels[mask], features[mask, :]


def remap_labels(y):
    """
    Rimappa le label secondo la logica desiderata.
    Attuale (come nel tuo codice):
      0 -> 0
      5,6 -> 1
      1,2,3,4 restano come sono (1..4)
    """
    y_new = np.copy(y)

    y_new[np.isin(y, [0])] = 0
    y_new[np.isin(y, [5, 6])] = 1
    # se vuoi davvero 3 classi (0, 1-4, 5-6) metti 2 qui sopra

    return y_new

def equalize_labels_temporal(labels, features):
    """
    Equalizza le classi con undersampling temporale uniforme.
    labels:   (N,)    array di etichette
    features: (N,F)   array di feature
    Ritorna: labels_eq, features_eq
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_count = counts.min()

    keep_indices = []

    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]     # indici di questa classe
        n = len(idx)

        # determinare il fattore di sottocampionamento
        # vogliamo ottenere esattamente min_count valori
        # formula: prendi circa ogni k-esimo campione
        k = n / min_count                    # es. 100 / 50 = 2 → tieni 1 ogni 2
        selected = []

        acc = 0.0
        for i in idx:
            if acc >= 1.0:
                selected.append(i)
                acc -= 1.0
            acc += 1.0 / k

        # Se per arrotondamenti abbiamo meno elementi del dovuto, pad finale
        if len(selected) < min_count:
            selected.extend(idx[len(selected):min_count])

        keep_indices.extend(selected)

    keep_indices = np.array(sorted(keep_indices))

    return labels[keep_indices], features[keep_indices]


def build_cv_sets(dataset_list):
    """
    dataset_list = [
        {"features": X0, "labels": y0},
        {"features": X1, "labels": y1},
        ...
    ]

    Output:
        cv_sets = list of 5 tuples:
           (X_train, y_train, X_test, y_test, fold_idx)
    """
    cv_sets = []

    for i in range(len(dataset_list)):
        test_ds = dataset_list[i]
        train_ds = [dataset_list[j] for j in range(len(dataset_list)) if j != i]

        # Concatenate training sets
        X_train = np.concatenate([ds["features"] for ds in train_ds], axis=0)
        y_train = np.concatenate([ds["labels"]   for ds in train_ds], axis=0)

        # Test set
        X_test = test_ds["features"]
        y_test = test_ds["labels"]

        cv_sets.append((X_train, y_train, X_test, y_test, i))  # include fold index

    return cv_sets


# Load the data
if feature_type == "ar":
    print("Loading AR coefficients for AR feature pipeline...")

    # Dataset paths
    if test == "pressure":
        dataset_paths = [
            os.path.join(script_dir, "data", "records_final", f"{finger}_{test}", "pressure", f"pressure_{i}")
            for i in range(5)
        ]

        AR_data = {}

        for order in AR_orders:
            AR_data[order] = {}
            for wl in AR_window_lengths:
                AR_data[order][wl] = {}
                for ov in AR_overlaps:

                    print(f"\n=== Loading AR sets for order={order}, wl={wl}, overlap={ov} ===")
                    AR_data[order][wl][ov] = []

                    # Convert overlap percent --> actual overlap fraction
                    overlap_fraction = ov / 100.0

                    for d, ds_path in enumerate(dataset_paths):

                        try:
                            # Load AR features and RAW labels (labels_full)
                            AR_features, labels_full = load_ar_set(ds_path, order, wl, ov)

                            N_windows = AR_features.shape[0]

                            # === 1. Ricostruisci le finestre label per questa combinazione
                            valid_idx, window_labels = window_labels_for_ar(
                                labels_full,
                                window_length=wl,
                                overlap=overlap_fraction
                            )

                            if len(window_labels) != N_windows:
                                raise ValueError(
                                    f"Mismatch windows AR vs label windows: {N_windows} vs {len(window_labels)}"
                                )

                            # === 2. Filtra AR features e labels per finestre pure
                            AR_features_valid = AR_features[valid_idx]
                            labels_valid      = window_labels[valid_idx]
                            #plot labels_valid
                            # plt.figure(figsize=(10, 4))
                            # plt.plot(labels_valid, label=f'Order {order}, WL {wl}, OV {ov}, Dataset {d}')
                            # plt.xlabel("Samples")
                            # plt.ylabel("Class")
                            # plt.title("Class Labels for Valid AR Windows")
                            # plt.legend()
                            # plt.grid()
                            # plt.show()
                            # Append this dataset for CV
                            AR_data[order][wl][ov].append({
                                "features": AR_features_valid,
                                "labels": labels_valid
                            })
                            if (len(AR_features_valid) != len(labels_valid)):
                                raise ValueError(
                                    f"ERROR Mismatch after filtering: {len(AR_features_valid)} features vs {len(labels_valid)} labels"
                                )
                            print(f"Loaded AR set OK: dataset pressure_{d}, valid windows = {len(valid_idx)}")

                        except FileNotFoundError:
                            print(f"Missing AR set: {order}_{wl}_{ov} in pressure_{d}")


                #plot the first label
                # print("ar data shape", AR_data[order][wl][ov][0]['features'].shape)
                # plt.figure(figsize=(10, 4))
                # plt.plot(AR_data[order][wl][ov][0]['features'], label=f'Order {order}, WL {wl}, OV {ov}')
                # plt.xlabel("Samples")
                # plt.ylabel("Class")
                # plt.title("Class Labels for First Dataset")
                # plt.legend()
                # plt.grid()
                # plt.show()
    elif test == "sliding":
        AR_coeffs_full  = np.load(os.path.join(script_dir, "data", "records_final", f'{finger}_level_3', "arCoeffs.npy"))
        noiseVar_full   = np.load(os.path.join(script_dir, "data", "records_final", f'{finger}_level_3', "noiseVariances.npy"))
        labels_full     = np.load(os.path.join(script_dir, "data", "records_final", f'{finger}_level_3', "labels_concatenated.npy"))
    else:
        raise ValueError("Invalid test type. Choose 'pressure' or 'sliding'.")
        
    param_combinations = []

    for order in AR_orders:
        for wl in AR_window_lengths:
            for ov in AR_overlaps:

                # ---------------------------
                # 1) Pre-process each dataset
                # ---------------------------
                processed_list = []

                for ds in AR_data[order][wl][ov]:
                    X = ds["features"]
                    y = ds["labels"]

                    # 1. POP unwanted labels
                    if pop_values_toggle:
                        y, X = pop_values(y, X, labels_to_pop)

                    # 2. REMAP labels
                    if remap_labels_toggle:
                        y = remap_labels(y)

                    if equalize_labels_toggle:
                        y, X = equalize_labels_temporal(y, X)

                    # 3. Skip empty datasets (could happen after POP)
                    if len(y) == 0:
                        print(f"[WARNING] Dataset empty after pop/remap for order={order}, wl={wl}, ov={ov}")
                        continue

                    processed_list.append({"features": X, "labels": y})

                # Replace original list with processed list
                AR_data[order][wl][ov] = processed_list

                # If less than 2 datasets left, cannot do CV
                if len(processed_list) < 2:
                    print(f"[ERROR] Not enough datasets for CV after preprocessing: order={order}, wl={wl}, ov={ov}")
                    continue

                # ---------------------------
                # 2) Build 5-fold CV sets
                # ---------------------------
                cv_sets = build_cv_sets(processed_list)

                # ---------------------------
                # 3) Generate SVM param sets
                # ---------------------------
                for (X_train, y_train, X_test, y_test, fold_idx) in cv_sets:
                    for C in C_values:
                        for gamma in gamma_values:
                            for kernel in kernels:

                                param_combinations.append({
                                    "feature_type": "ar",
                                    "order": order,
                                    "wl": wl,
                                    "ov": ov,
                                    "fold": fold_idx,
                                    "C": C,
                                    "gamma": gamma,
                                    "kernel": kernel,
                                    "X_train": X_train,
                                    "y_train": y_train,
                                    "X_test": X_test,
                                    "y_test": y_test
                                })
    for i, pc in enumerate(param_combinations[:50]):  # solo primi 20
        print(f"#{i}: order={pc['order']}, wl={pc['wl']}, ov={pc['ov']}, "
            f"fold={pc['fold']}, C={pc['C']}, gamma={pc['gamma']}, kernel={pc['kernel']}, "
            f"X_train={pc['X_train'].shape}, X_test={pc['X_test'].shape}")



    piezo_values  = features_full[:N_train]
    label_points  = labels_full[:N_train]

    piezo_values2 = features_full[N_train:]
    label_points2 = labels_full[N_train:]

    print("AR Train:", piezo_values.shape, label_points.shape)
    print("AR Test: ", piezo_values2.shape, label_points2.shape)
else:
    piezo_values = np.load(piezo_data_path)
    label_points = np.load(labels_data_path)
    piezo_values2 = np.load(piezo_data_path_2)
    label_points2 = np.load(labels_data_path_2)
    print("Set 1 - Labels shape:", label_points.shape)
    print("Set 2 - Labels shape:", label_points2.shape)
    print("Set 1 - Piezo values shape:", piezo_values.shape)



# Optional: Plot class labels for both sets
plt.figure(figsize=(10, 4))
plt.plot(label_points, label="Set 1")
plt.plot(label_points2, label="Set 2")
plt.xlabel("Samples")
plt.ylabel("Class")
plt.title("Class Labels")
plt.legend()
plt.grid()
plt.show()



if pop_values_toggle:
    label_points,  piezo_values  = pop_values(label_points,  piezo_values,  labels_to_pop)
    label_points2, piezo_values2 = pop_values(label_points2, piezo_values2, labels_to_pop)


# Remap labels into 3 classes:
# 0 → 0
# 1,2,3,4 → 1
# 5,6 → 2


if remap_labels_toggle:
    label_points  = remap_labels(label_points)
    label_points2 = remap_labels(label_points2)



plt.figure(figsize=(10, 4))
plt.plot(label_points, label="Set 1")
plt.plot(label_points2, label="Set 2")
plt.xlabel("Samples")
plt.ylabel("Class")
plt.title("Class Labels after remapping")
plt.legend()
plt.grid()
plt.show()
plt.close('all')  # Close all previous figures

# ---------------------------------
# 1.5 Label Equalization
# ---------------------------------

def equalize_labels(label_points, piezo_values):
    """
    Equalize the number of samples for each class by undersampling (NO SHUFFLE).
    """
    unique_labels, counts = np.unique(label_points, return_counts=True)
    min_count = counts.min()

    final_indices = []

    for label in unique_labels:
        idx = np.where(label_points == label)[0]

        # undersample if necessary, but keep order
        if len(idx) > min_count:
            chosen = idx[:min_count]   # NO RANDOM, KEEP FIRST min_count
        else:
            chosen = idx

        final_indices.append(chosen)

    # keep chronological order
    final_indices = np.sort(np.concatenate(final_indices))

    return label_points[final_indices], piezo_values[final_indices]


if equalize_labels_toggle:
    print("len label_points", len(label_points))
    label_points, piezo_values = equalize_labels(label_points, piezo_values)
    label_points2, piezo_values2 = equalize_labels(label_points2, piezo_values2)
    print("len label_points after equalizing", len(label_points))

# ---------------------------------
# 2. Signal Segmentation & Labels
# ---------------------------------
def segment_signal(signal, fs, window_length, overlap):
    """Segment the signal into windows with the given length and overlap."""
    if overlap == 1.0:
        step = 1
    else:
        step = int(window_length * fs * (1 - overlap))  # step size (samples)
    window_size = int(window_length * fs)           # window size (samples)
    segments = []
    for start in range(0, len(signal) - window_size + 1, step):
        segments.append(signal[start : start + window_size])
    return np.array(segments)

from scipy.signal import butter, filtfilt
import numpy as np

def apply_filter(signal, fs, filter_type, freq=None, alpha=0.1):
    """
    Apply a highpass, lowpass, or exponential moving average (EMA) filter to the signal.
    
    Parameters:
        signal (array-like): Input signal.
        fs (float): Sampling frequency.
        filter_type (str): Type of filter ('highpass', 'lowpass', 'ema', or 'nothing').
        freq (float, optional): Cutoff frequency (required for 'highpass' and 'lowpass' filters).
        alpha (float, optional): Smoothing factor for EMA (default=0.1).
    
    Returns:
        array-like: Filtered signal.
    """
    order = 4  # Filter order for Butterworth filters
    nyquist = 0.5 * fs  # Nyquist frequency
    
    if filter_type in ["highpass", "lowpass"]:
        if freq is None:
            raise ValueError("Frequency (freq) must be provided for highpass and lowpass filters.")
        normal_cutoff = freq / nyquist  # Normalize cutoff frequency
        btype = "high" if filter_type == "highpass" else "low"
        b, a = butter(order, normal_cutoff, btype=btype, analog=False)
        return filtfilt(b, a, signal, axis=-1)  # Apply the filter
    
    elif filter_type == "ema":
        # Apply Exponential Moving Average (EMA)
        ema_signal = np.zeros_like(signal)
        ema_signal[0] = signal[0]  # Initialize with the first value
        for i in range(1, len(signal)):
            ema_signal[i] = alpha * signal[i] + (1 - alpha) * ema_signal[i - 1]
        return ema_signal

    return signal  # Return original signal if filter_type is "nothing"

def find_threshold(signal):
#return the first and third quartile of the signal 1000xn
    first_quartile = np.percentile(signal, 0, axis=0)*2
    third_quartile = np.percentile(signal, 100, axis=0) *2
    print("first quartile", first_quartile)
    print("third quartile", third_quartile)
    return np.array([first_quartile, third_quartile])


#i want to print the number of labels appearing
print("number of labels appearing", np.unique(label_points, return_counts=True))
# Create segments for both datasets
if feature_type not in ["nothing", "integral", "ar"]:
    segments = {
        wl: {
            ov: apply_filter(segment_signal(piezo_values, fs, wl, ov), fs, filter_type, freq) 
            for ov in overlaps
        } 
        for wl in window_lengths
    }
    segments2 = {
        wl: {
            ov: apply_filter(segment_signal(piezo_values2, fs, wl, ov), fs, filter_type, freq) 
            for ov in overlaps
        } 
        for wl in window_lengths
    }
elif filter_type != "nothing":
    # Apply filtering WITHOUT segmentation
    piezo_values = apply_filter(piezo_values, fs, filter_type, freq)
    piezo_values2 = apply_filter(piezo_values2, fs, filter_type, freq)
    #find threshold using the first 1000 samples for each column
    threshold = find_threshold(piezo_values[:2000])

    threshold2 = find_threshold(piezo_values2[:2000])

# Assign labels to segments for both datasets
labeled_segments = {}
labeled_segments2 = {}
valid_segments = {}
valid_segments2 = {}

if feature_type not in ["nothing", "integral", "ar"]:
    for wl in window_lengths:
        labeled_segments[wl] = {}
        labeled_segments2[wl] = {}
        valid_segments[wl] = {}
        valid_segments2[wl] = {}
        for ov in overlaps:
            print(f"\nProcessing Set 1 - Window Length: {wl}s, Overlap: {ov}")
            valid_segs, seg_labels = assign_labels_to_segments(segments[wl][ov], fs, wl, ov, label_points)
            valid_segments[wl][ov] = valid_segs
            labeled_segments[wl][ov] = seg_labels
            
            print(f"\nProcessing Set 2 - Window Length: {wl}s, Overlap: {ov}")
            valid_segs2, seg_labels2 = assign_labels_to_segments(segments2[wl][ov], fs, wl, ov, label_points2)
            valid_segments2[wl][ov] = valid_segs2
            labeled_segments2[wl][ov] = seg_labels2
# ---------------------------------
# 3. Feature Extraction
# ---------------------------------
def extract_wavelet_features(segments, wavelet, level=4):
    """Extract wavelet-based features from each segment."""
    features = []
    for segment in segments:
        coeffs = pywt.wavedec(segment, wavelet, level=level, axis=0)
        # Compute marginal coefficients (skip the approximation coefficients)
        marginal = [np.sum(np.abs(c), axis=0) for c in coeffs[1:]]
        features.append(np.concatenate(marginal))

    return np.array(features)

def extract_stft_features(segments, fs, nperseg=64, noverlap=4):
    """Extract STFT-based features from each segment."""
    features = []
    for segment in segments:
        f, t, Zxx = stft(segment, fs=fs, nperseg=nperseg, noverlap=noverlap)
        # Take the magnitude of the STFT and flatten it
        features.append(np.abs(Zxx).flatten())
    return np.array(features)


# Extract features based on the selected feature_type
if feature_type == "wavelet":
    features = {
        wl: {ov: {w: extract_wavelet_features(valid_segments[wl][ov], w) for w in wavelets} for ov in overlaps}
        for wl in window_lengths
    }
    features2 = {
        wl: {ov: {w: extract_wavelet_features(valid_segments2[wl][ov], w) for w in wavelets} for ov in overlaps}
        for wl in window_lengths
    }

elif feature_type == "stft":
    features = {
        wl: {ov: extract_stft_features(valid_segments[wl][ov], fs) for ov in overlaps}
        for wl in window_lengths
    }
    features2 = {
        wl: {ov: extract_stft_features(valid_segments2[wl][ov], fs) for ov in overlaps}
        for wl in window_lengths
    }

elif feature_type == "integral":
    # Compute the integral of piezo_values
    if treshold:
        plt.figure(figsize=(10, 4))
        plt.plot(piezo_values)
        plt.xlabel("Samples")
        plt.ylabel("Value")
        plt.title("Piezo Values")
        plt.legend()
        plt.grid()
        plt.show()
    #sum only where the signal is out of threshold
        features = np.cumsum(np.where((piezo_values < threshold[0]) | (piezo_values > threshold[1]), piezo_values, 0), axis=0) / fs
        features2 = np.cumsum(np.where((piezo_values2 < threshold2[0]) | (piezo_values2 > threshold2[1]), piezo_values2, 0), axis=0) / fs
    else:
        features = np.cumsum(piezo_values, axis=0)/fs
        features2 = np.cumsum(piezo_values2, axis=0)/fs
    plt.figure(figsize=(10, 4))
    plt.plot(features)
    plt.xlabel("Samples")
    plt.ylabel("Integral")
    plt.title("Integral of Piezo Values")
    plt.legend()
    plt.grid()
    plt.show()

elif feature_type == "nothing":
    # Directly use the full signal without feature extraction
    features = piezo_values
    features2 = piezo_values2

elif feature_type == "ar":
    # ------------------------------------------------------------
    # AR features & labels already aligned and split above
    # ------------------------------------------------------------
    features  = piezo_values      # [N_train × 88]
    features2 = piezo_values2     # [N_test  × 88]

    labels_train = label_points   # [N_train]
    labels_test  = label_points2  # [N_test]

    # Build labeled matrices
    labeled_features  = np.hstack((features,  labels_train[:, None]))
    labeled_features2 = np.hstack((features2, labels_test[:, None]))

    print("AR labeled_features :", labeled_features.shape)
    print("AR labeled_features2:", labeled_features2.shape)


else:
    raise ValueError("Invalid type. Choose 'wavelet', 'stft', 'integral', 'nothing', or 'ar'.")

# Print the shapes for verification
if feature_type == "nothing":
    print(f"Using raw signals directly. Shape: {features.shape}")
elif feature_type == "integral":
    print(f"Using integral values. Shape: {features.shape}")
else:
    for wl in window_lengths:
        for ov in overlaps:
            if feature_type == "wavelet":
                for w in wavelets:
                    print(f"Window: {wl}s, Overlap: {ov}, Wavelet: {w}, Shape: {features[wl][ov][w].shape}")
            elif feature_type == "stft":
                print(f"Window: {wl}s, Overlap: {ov}, Shape: {features[wl][ov].shape}")

# ---------------------------------
# 4. Combine Features and Labels
# ---------------------------------
labeled_features = {}
labeled_features2 = {}
if feature_type in ["nothing", "integral", "ar"]:
    # Directly pair raw signals with labels
    if len(label_points) != features.shape[0]:
        raise ValueError(
            f"Mismatch in raw signal: {features.shape[0]} samples vs {len(label_points)} labels."
        )
    labeled_features = np.hstack((features, label_points[:, None]))
    labeled_features2 = np.hstack((features2, label_points2[:, None]))

else:
    for wl in window_lengths:
        labeled_features[wl] = {}
        labeled_features2[wl] = {}
        for ov in overlaps:
            if feature_type == "wavelet":
                labeled_features[wl][ov] = {}
                labeled_features2[wl][ov] = {}
                for w in wavelets:
                    feature_matrix = features[wl][ov][w]
                    feature_matrix2 = features2[wl][ov][w]
                    if len(labeled_segments[wl][ov]) != feature_matrix.shape[0]:
                        raise ValueError(
                            f"Mismatch for window {wl}s, overlap {ov}, wavelet {w}: "
                            f"{feature_matrix.shape[0]} features vs {len(labeled_segments[wl][ov])} labels."
                        )
                    # Append labels as the last column
                    labeled_feature_matrix = np.hstack((feature_matrix, labeled_segments[wl][ov][:, None]))
                    labeled_feature_matrix2 = np.hstack((feature_matrix2, labeled_segments2[wl][ov][:, None]))
                    labeled_features[wl][ov][w] = labeled_feature_matrix
                    labeled_features2[wl][ov][w] = labeled_feature_matrix2
            elif feature_type == "stft":
                feature_matrix = features[wl][ov]
                feature_matrix2 = features2[wl][ov]
                if len(labeled_segments[wl][ov]) != feature_matrix.shape[0]:
                    raise ValueError(
                        f"Mismatch for window {wl}s, overlap {ov}: "
                        f"{feature_matrix.shape[0]} features vs {len(labeled_segments[wl][ov])} labels."
                    )
                # Append labels as the last column
                labeled_feature_matrix = np.hstack((feature_matrix, labeled_segments[wl][ov][:, None]))
                labeled_feature_matrix2 = np.hstack((feature_matrix2, labeled_segments2[wl][ov][:, None]))
                labeled_features[wl][ov] = labeled_feature_matrix
                labeled_features2[wl][ov] = labeled_feature_matrix2

# ---------------------------------
# 5. SVM Grid Search with Parallelism
# ---------------------------------


def get_color_code(text):
    """Generate a unique color for each hyperparameter combination."""
    colors = [
        "\033[91m",  # Red
        "\033[92m",  # Green
        "\033[93m",  # Yellow
        "\033[94m",  # Blue
        "\033[95m",  # Magenta
        "\033[96m",  # Cyan
    ]
    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)  # Generate a hash
    return colors[hash_val % len(colors)]  # Pick a color cyclically

def train_and_evaluate_svm(wl, w, ov, C, gamma, kernel, X_train, y_train, X_test, y_test):
    """Train an SVM and evaluate its accuracy."""
    if feature_type == "wavelet":
        param_set = f"Window: {wl}s, Wavelet: {w}, Overlap: {ov}, Kernel: {kernel}, Gamma: {gamma}, C: {C}"
    elif feature_type == "stft":
        param_set = f"Window: {wl}s, Overlap: {ov}, Kernel: {kernel}, Gamma: {gamma}, C: {C}"
    elif feature_type == "nothing":
        param_set = f"Raw Signal, Kernel: {kernel}, Gamma: {gamma}, C: {C}"
    elif feature_type == "integral":
        param_set = f"Integral Signal, Kernel: {kernel}, Gamma: {gamma}, C: {C}"
    elif feature_type == "ar":
        param_set = f"AR Features, Kernel: {kernel}, Gamma: {gamma}, C: {C}"
    color = get_color_code(param_set)
    
    print(f"{color}[{time.strftime('%H:%M:%S')}] Started training with: {param_set}\033[0m")
    
    start_time = time.time()  # Track training time
    
    clf = SVC(kernel=kernel, gamma=gamma, C=C)
    clf.fit(X_train, y_train)
    
    training_time = time.time() - start_time  # Compute training duration
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    #accuracy = precision_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"{color}[{time.strftime('%H:%M:%S')}] Finished training with: {param_set} (Accuracy: {accuracy:.4f}) - Took {training_time:.2f}s\033[0m")
    
    # Return results based on the feature_type
    return {
        "window_length": wl if feature_type not in ["nothing", "integral"] else None,
        "wavelet": w if feature_type == "wavelet" else None,
        "overlap": ov if feature_type not in ["nothing", "integral"] else None,
        "C": C,
        "gamma": gamma,
        "kernel": kernel,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "training_time": training_time,
        "model": clf
    }

def train_and_evaluate_hmm(wl, w, ov, X_train, y_train, X_test, y_test, unsupervised=True, initial_start_prob=None, initial_transition_matrix=None, n_states=2):
    """
    Train an HMM and evaluate its accuracy.
    
    Parameters:
    - wl: Window length (used for logging)
    - w: Wavelet feature_type (used for logging)
    - ov: Overlap (used for logging)
    - X_train: Training features
    - y_train: Training labels (ignored if unsupervised=True)
    - X_test: Test features
    - y_test: Test labels (ignored if unsupervised=True)
    - unsupervised: If True, use unsupervised training with a custom transition matrix
    - initial_transition_matrix: Custom transition matrix for unsupervised training
    - n_states: Maximum number of states to consider for grid search (only used in supervised mode)
    """
    if unsupervised:
        n_states = initial_transition_matrix.shape[0]
        
    if feature_type == "wavelet":
        param_set = f"Window: {wl}s, Wavelet: {w}, Overlap: {ov}, n_states: {n_states}"
    elif feature_type == "stft":
        param_set = f"Window: {wl}s, Overlap: {ov}, n_states: {n_states}"
    elif feature_type == "nothing":
        param_set = f"Raw Signal, n_states: {n_states}"
    elif feature_type == "integral":
        param_set = f"Integral Signal, n_states: {n_states}"
    color = get_color_code(param_set)
    
    print(f"{color}[{time.strftime('%H:%M:%S')}] Started training with: {param_set}\033[0m")
    
    start_time = time.time()  # Track training time

    if unsupervised:
        # Unsupervised training with custom transition matrix
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            params="tmc",
            init_params='mc',  # Do not re-initialize any parameters
        )
        model.startprob_=initial_start_prob
        model.transmat_=initial_transition_matrix  # Set the custom transition matrix
        model.fit(X_train)  # Train on the entire dataset without labels
    else:
        # Supervised training (one HMM per class)
        unique_labels = np.unique(y_train)
        best_accuracy = -np.inf  # Track the best overall accuracy
        best_models = {}  # Store the best models for each label
        best_n_states = {}  # Store the best n_states for each label
        
        # Generate all possible combinations of n_states for each label
        n_states_range = range(2, n_states + 1)
        n_states_combinations = list(product(n_states_range, repeat=len(unique_labels)))
        
        print(f"{color}[{time.strftime('%H:%M:%S')}] Total combinations to evaluate: {len(n_states_combinations)}\033[0m")
        
        # Evaluate each combination of n_states
        for combination in n_states_combinations:
            models = {}
            combination_n_states = dict(zip(unique_labels, combination))  # Map labels to n_states
            
            # Train models for this combination
            for label in unique_labels:
                model = hmm.GaussianHMM(
                    n_components=combination_n_states[label],
                    covariance_type="diag",
                    n_iter=100
                )
                model.fit(X_train[y_train == label])
                models[label] = model
            
            # Evaluate the combination on the validation set (or training set if no validation set)
            y_pred = []
            for sample in X_train:  # Use X_train for validation (or split into train/validation sets)
                scores = [model.score(sample.reshape(1, -1)) for model in models.values()]
                y_pred.append(unique_labels[np.argmax(scores)])
            
            y_pred = np.array(y_pred)
            accuracy = accuracy_score(y_train, y_pred)
            print(f"{color}[{time.strftime('%H:%M:%S')}] Combination: {combination} - Accuracy: {accuracy:.4f}\033[0m")
            # Update the best combination if this one is better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_models = models
                best_n_states = combination_n_states
                print(f"{color}[{time.strftime('%H:%M:%S')}] New best combination: {best_n_states} (Accuracy: {best_accuracy:.4f})\033[0m")
        
        print(f"{color}[{time.strftime('%H:%M:%S')}] Best n_states combination: {best_n_states} (Accuracy: {best_accuracy:.4f})\033[0m")
    
    training_time = time.time() - start_time
    
    # Evaluate the model
    if unsupervised:
        # Predict hidden states for unsupervised training
        hidden_states_train = model.predict(X_train)
        hidden_states_test = model.predict(X_test)
        
        # Since this is unsupervised, we can't compute accuracy directly
        # Instead, we can analyze the hidden states or use clustering metrics
        print(f"{color}[{time.strftime('%H:%M:%S')}] Finished unsupervised training with: {param_set} - Took {training_time:.2f}s\033[0m")
        
        # Return hidden states and model
        return {
            "window_length": wl if feature_type not in ["nothing", "integral"] else None,
            "wavelet": w if feature_type == "wavelet" else None,
            "overlap": ov if feature_type not in ["nothing", "integral"] else None,
            "n_states": n_states,
            "hidden_states_train": hidden_states_train,
            "hidden_states_test": hidden_states_test,
            "training_time": training_time,
            "model": model
        }
    else:
        # Supervised evaluation on the test set
        y_pred = []
        for sample in X_test:
            scores = [model.score(sample.reshape(1, -1)) for model in best_models.values()]
            y_pred.append(unique_labels[np.argmax(scores)])
        
        y_pred = np.array(y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"{color}[{time.strftime('%H:%M:%S')}] Finished training with: {param_set} (Accuracy: {accuracy:.4f}) - Took {training_time:.2f}s\033[0m")
        
        return {
            "window_length": wl if feature_type not in ["nothing", "integral"] else None,
            "wavelet": w if feature_type == "wavelet" else None,
            "overlap": ov if feature_type not in ["nothing", "integral"] else None,
            "n_states": best_n_states,  # Return the best n_states for each label
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "training_time": training_time,
            "models": best_models
        }
    
# Prepare all parameter combinations (each tuple contains all required data)
param_combinations = []
if classifier == "svm":
    if feature_type in ["nothing", "integral", "ar"]:
        # AR behaves like nothing/integral → simple train/test split already done
        for C in C_values:
            for gamma in gamma_values:
                for kernel in kernels:
                    param_combinations.append(
                        (None, None, None, C, gamma, kernel,
                         labeled_features[:, :-1], labeled_features[:, -1],
                         labeled_features2[:, :-1], labeled_features2[:, -1])
                    )

    else:
        # WAVELET or STFT
        for wl in window_lengths:
            for ov in overlaps:
                if feature_type == "wavelet":
                    for w in wavelets:
                        for C in C_values:
                            for gamma in gamma_values:
                                for kernel in kernels:
                                    param_combinations.append(
                                        (wl, w, ov, C, gamma, kernel,
                                         labeled_features[wl][ov][w][:, :-1],   labeled_features[wl][ov][w][:, -1],
                                         labeled_features2[wl][ov][w][:, :-1],  labeled_features2[wl][ov][w][:, -1])
                                    )

                elif feature_type == "stft":
                    for C in C_values:
                        for gamma in gamma_values:
                            for kernel in kernels:
                                param_combinations.append(
                                    (wl, None, ov, C, gamma, kernel,
                                     labeled_features[wl][ov][:, :-1],   labeled_features[wl][ov][:, -1],
                                     labeled_features2[wl][ov][:, :-1],  labeled_features2[wl][ov][:, -1])
                                )

if classifier == "hmm":
    if feature_type  in ["nothing", "integral"]:
        for n_states in n_states_values:
            param_combinations.append(
                (None, None, None, None, None, None,  # HMM does not use C, gamma, kernel
                 labeled_features[:, :-1], labeled_features[:, -1],
                 labeled_features2[:, :-1], labeled_features2[:, -1],
                 n_states)
            )
    else:
        for wl in window_lengths:
            for ov in overlaps:
                if feature_type == "wavelet":
                    for w in wavelets:
                        for n_states in n_states_values:
                            param_combinations.append(
                                (wl, w, ov, None, None, None,
                                 labeled_features[wl][ov][w][:, :-1], labeled_features[wl][ov][w][:, -1],
                                 labeled_features2[wl][ov][w][:, :-1], labeled_features2[wl][ov][w][:, -1],
                                 n_states)
                            )
                elif feature_type == "stft":
                    for n_states in n_states_values:
                        param_combinations.append(
                            (wl, None, ov, None, None, None,
                             labeled_features[wl][ov][:, :-1], labeled_features[wl][ov][:, -1],
                             labeled_features2[wl][ov][:, :-1], labeled_features2[wl][ov][:, -1],
                             n_states)
                        )


# Use joblib to process the SVM training in parallel
num_cores = min(20, multiprocessing.cpu_count())
if classifier == "svm":
    tasks = [delayed(train_and_evaluate_svm)(*params) for params in param_combinations]
elif classifier == "hmm":
    tasks = [delayed(train_and_evaluate_hmm)(*params[:3], params[6], params[7], params[8], params[9], unsupervised=unsupervised, initial_start_prob=initial_start_prob, initial_transition_matrix=initial_transition_matrix, n_states=params[10]) for params in param_combinations]    
results_test_on_set2 = Parallel(n_jobs=num_cores)(tqdm(tasks, total=len(tasks), desc="Processing"))

if classifier == "svm":
    best_result_test_on_set2 = max(results_test_on_set2, key=lambda x: x["accuracy"])
    sorted_results = sorted(results_test_on_set2, key=lambda x: x["accuracy"], reverse=True)
elif classifier == "hmm":
    # Use accuracy if available, otherwise fall back on likelihood score
    best_result_test_on_set2 = max(results_test_on_set2, key=lambda x: x.get("accuracy", 0))
    sorted_results = sorted(results_test_on_set2, key=lambda x: x.get("accuracy", 0), reverse=True)

for result in results_test_on_set2:
    if classifier == "svm":
        if feature_type == "wavelet":
            print(f"Window: {result['window_length']}s, Wavelet: {result['wavelet']}, Overlap: {result['overlap']}, "
                  f"Kernel: {result['kernel']}, Gamma: {result['gamma']}, C: {result['C']}, "
                  f"accuracy: {result['accuracy']:.4f}")
        elif feature_type == "stft":
            print(f"Window: {result['window_length']}s, Overlap: {result['overlap']}, "
                  f"Kernel: {result['kernel']}, Gamma: {result['gamma']}, C: {result['C']}, "
                  f"accuracy: {result['accuracy']:.4f}")
        elif feature_type == "nothing":
            print(f"Raw Signal, Kernel: {result['kernel']}, Gamma: {result['gamma']}, C: {result['C']}, "
                  f"accuracy: {result['accuracy']:.4f}")
        elif feature_type == "integral":
            print(f"Integral Signal, Kernel: {result['kernel']}, Gamma: {result['gamma']}, C: {result['C']}, "
                  f"accuracy: {result['accuracy']:.4f}")
    elif classifier == "hmm":
        print(f"Window: {result['window_length']}s, Overlap: {result['overlap']}, n_states: {result['n_states']} "
              f"Accuracy: {result.get('accuracy', 'N/A')}")

print("\nTop 30 Combinations (Test on Set 2):")
for i, result in enumerate(sorted_results[:30], start=1):
    if classifier == "svm":
        if feature_type == "wavelet":
            print(f"Rank {i}: Window: {result['window_length']}s, Wavelet: {result['wavelet']}, "
                  f"Overlap: {result['overlap']}, Kernel: {result['kernel']}, Gamma: {result['gamma']}, C: {result['C']}, "
                  f"accuracy: {result['accuracy']:.4f}")
        elif feature_type == "stft":
            print(f"Rank {i}: Window: {result['window_length']}s, "
                  f"Overlap: {result['overlap']}, Kernel: {result['kernel']}, Gamma: {result['gamma']}, C: {result['C']}, "
                  f"accuracy: {result['accuracy']:.4f}")
        elif feature_type == "nothing":
            print(f"Rank {i}: Raw Signal, Kernel: {result['kernel']}, Gamma: {result['gamma']}, C: {result['C']}, "
                  f"accuracy: {result['accuracy']:.4f}")
        elif feature_type == "integral":
            print(f"Rank {i}: Integral Signal, Kernel: {result['kernel']}, Gamma: {result['gamma']}, C: {result['C']}, "
                  f"accuracy: {result['accuracy']:.4f}")
        elif feature_type == "ar":
            print(f"Rank {i}: AR Features, Kernel: {result['kernel']}, Gamma: {result['gamma']}, C: {result['C']}, "
                  f"accuracy: {result['accuracy']:.4f}")
    elif classifier == "hmm":
        print(f"Rank {i}: Window: {result['window_length']}s, Overlap: {result['overlap']}, "
              f"Accuracy: {result.get('accuracy', 'N/A')}")

if classifier == "svm":
    best_cm = best_result_test_on_set2["confusion_matrix"]
elif classifier == "hmm" and "confusion_matrix" in best_result_test_on_set2:
    best_cm = best_result_test_on_set2["confusion_matrix"]
else:
    best_cm = None  # No confusion matrix for unsupervised HMM

# Extract unique labels
if feature_type in ["nothing", "integral", "ar"]:
    unique_labels = np.unique(labeled_features2[:, -1])
else:
    if feature_type == "wavelet":
        wl = best_result_test_on_set2["window_length"]
        ov = best_result_test_on_set2["overlap"]
        w = best_result_test_on_set2["wavelet"]
        unique_labels = np.unique(labeled_features2[wl][ov][w][:, -1])
    elif feature_type == "stft":
        wl = best_result_test_on_set2["window_length"]
        ov = best_result_test_on_set2["overlap"]
        unique_labels = np.unique(labeled_features2[wl][ov][:, -1])

# Plot Confusion Matrix if available
if best_cm is not None:
    best_cm_normalized = best_cm.astype('float') / best_cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(best_cm_normalized, annot=True, fmt=".2%", cmap="Blues", 
                     xticklabels=unique_labels, yticklabels=unique_labels,
                     cbar_kws={'label': 'Number of Classifications'})  # Add label to the color bar
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    # Add accuracy to the title
    if classifier == "svm":
        if feature_type == "nothing":
            plt.title(
                f"Confusion Matrix (Best: Raw Signal, Kernel: {best_result_test_on_set2['kernel']}, "
                f"Gamma: {best_result_test_on_set2['gamma']}, C: {best_result_test_on_set2['C']}, "
                f"Accuracy: {best_result_test_on_set2['accuracy']:.2%})"
            )
        elif feature_type == "integral":
            plt.title(
                f"Confusion Matrix (Best: Integral Signal, Kernel: {best_result_test_on_set2['kernel']}, "
                f"Gamma: {best_result_test_on_set2['gamma']}, C: {best_result_test_on_set2['C']}, "
                f"Accuracy: {best_result_test_on_set2['accuracy']:.2%})"
            )
        elif feature_type == "ar":
            plt.title(
                f"Confusion Matrix (Best AR Features, Kernel: {best_result_test_on_set2['kernel']}, "
                f"Gamma: {best_result_test_on_set2['gamma']}, C: {best_result_test_on_set2['C']}, "
                f"Accuracy: {best_result_test_on_set2['accuracy']:.2%})"
            )

        else:
            plt.title(
                f"Confusion Matrix (Best: Window: {best_result_test_on_set2['window_length']}s, "
                f"Overlap: {best_result_test_on_set2['overlap']}, "
                f"Kernel: {best_result_test_on_set2['kernel']}, "
                f"Gamma: {best_result_test_on_set2['gamma']}, C: {best_result_test_on_set2['C']}, "
                f"Accuracy: {best_result_test_on_set2['accuracy']:.2%})"
            )
    elif classifier == "hmm":
        plt.title(
            f"Confusion Matrix (Best: Window: {best_result_test_on_set2['window_length']}s, "
            f"Overlap: {best_result_test_on_set2['overlap']}, "
            f"Accuracy: {best_result_test_on_set2.get('accuracy', 'N/A')})"
        )

    # Adjust the color bar to reflect raw counts, if it exists
    cbar = ax.collections[0].colorbar  # Get the color bar from the heatmap
    if cbar:
        tick_labels = cbar.get_ticks()  # Get current tick positions
        cbar.set_ticks(tick_labels)  # Ensure ticks are properly set
        cbar.set_ticklabels([f"{int(t * best_cm.sum())}" for t in tick_labels])  # Convert to raw counts

    plt.show()
# Extract the correct feature matrix for prediction
if feature_type in ["nothing", "integral", "ar"]:
    X_test = labeled_features2[:, :-1]  # Features are all columns except the last
    y_test = labeled_features2[:, -1]   # Labels are the last column
else:
    # For wavelet or STFT, use the best parameters from the results
    wl = best_result_test_on_set2["window_length"]
    ov = best_result_test_on_set2["overlap"]
    if feature_type == "wavelet":
        w = best_result_test_on_set2["wavelet"]
        X_test = labeled_features2[wl][ov][w][:, :-1]  # Features are all columns except the last
        y_test = labeled_features2[wl][ov][w][:, -1]   # Labels are the last column
    elif feature_type == "stft":
        X_test = labeled_features2[wl][ov][:, :-1]  # Features are all columns except the last
        y_test = labeled_features2[wl][ov][:, -1]   # Labels are the last column

# Get the predicted labels for the test set
if classifier == "svm":
    y_pred = best_result_test_on_set2["model"].predict(X_test)
elif classifier == "hmm":
    if unsupervised:
        y_pred = best_result_test_on_set2["hidden_states_test"]
    else:
        y_pred = []
        for sample in X_test:
            scores = [model.score(sample.reshape(1, -1)) for model in best_result_test_on_set2["models"].values()]
            y_pred.append(unique_labels[np.argmax(scores)])
        y_pred = np.array(y_pred)

# Plot the test set with predicted classes
plt.figure(figsize=(10, 6))

for label in np.unique(y_pred):
    plt.scatter(
        np.where(y_pred == label)[0],  # X-axis: sample indices
        X_test[y_pred == label].mean(axis=1),  # Y-axis: mean feature value (or any other feature)
        label=f"Predicted Class {label}",
        alpha=0.6,
        s=0.5
    )
plt.plot(y_test, label="True Class", color="black", linestyle="--")
plt.xlabel("Sample Index")
plt.ylabel("Feature Value (Mean)")
plt.title("Test Set Classification Visualization")
plt.legend()
plt.grid()
plt.show()
if "model" in best_result_test_on_set2:
    joblib.dump(best_result_test_on_set2["model"], best_model_path)
    print(f"Saved single model to {best_model_path}")
elif "models" in best_result_test_on_set2:
    joblib.dump(best_result_test_on_set2["models"], best_model_path)
    print(f"Saved multiple models to {best_model_path}")
else:
    print("Error: Neither 'model' nor 'models' found in best_result_test_on_set2")