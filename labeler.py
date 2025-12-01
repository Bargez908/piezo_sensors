import numpy as np
import matplotlib.pyplot as plt
import os

# Creazione di un semplice grafico
script_dir = os.path.dirname(os.path.abspath(__file__))

finger="thumb"
test_n = 4
folder=f"sliding_{finger}_{test_n}"
test = f"{finger}_level_3\\sliding"

macro_folder = "records_final"
test_type = "sliding"  # "sliding" or "pressure"

folder1 = f'sliding_{test_n}'

# Define the relative path to the file
piezo_data_path = os.path.join(script_dir,"data\\records_final", test, folder1, f'{finger}_downsampled.npy')
trigger_data_path = os.path.join(script_dir,"data\\records_final", test, folder1, 'triggers_downsampled.npy')

sensor_values_path = os.path.join(script_dir,"data\\records_final", test, folder1, 'sensor_values_downsampled.npy')
labels_path = os.path.join(script_dir,"data\\records_final", test, folder1, 'labels.npy')   #pressure levels
#labels_path2 = os.path.join(script_dir,"data\\records_final", test, folder1, 'labels2.npy') #pressure/no pressure
# Load the data
piezo_values = np.load(piezo_data_path)  # Finger sensor data
sensor_values = np.load(sensor_values_path)  # Force sensor data
trigger_values = np.load(trigger_data_path)  # Trigger data


# Ignore the last row (time values)
trigger_values_data = trigger_values[:-1, :]

# Find indices where values change from 0 to 1
diff = np.diff(trigger_values_data, axis=1)
print("Diff shape:", diff.shape)
# Find indices where the difference is non-zero
change_indices = np.where(diff != 0)
# Adjust the column indices to match the original array
# Flatten indices into a single array
all_change_indices = np.sort(np.unique(change_indices[1]+1))
all_change_indices = np.concatenate(([0], all_change_indices, [len(trigger_values_data.T)]))

diffs = np.diff(all_change_indices)

# Keep indices where the difference is greater than 1
all_change_indices = all_change_indices[np.concatenate(([True], diffs > 1))]

# Print results
print("All change indices:", all_change_indices)
print(f"Initial trigger data shape: {trigger_values.shape}")

trigger_values_cleaned = []
sensor_values_cleaned = []
piezo_values_cleaned = []

for i in range(len(all_change_indices) - 1):
    start_idx = all_change_indices[i]
    end_idx = all_change_indices[i + 1]

    # Calculate the window size and the slicing indices
    window_size = end_idx - start_idx
    start_cut = start_idx + window_size // 6 * 2  # Skip the first 2/6
    end_cut = end_idx - window_size // 6      # Skip the last 1/6

    # Extract the cleaned segment
    trigger_values_cleaned.append(trigger_values[:, start_cut:end_cut])
    sensor_values_cleaned.append(sensor_values[start_cut:end_cut])
    piezo_values_cleaned.append(piezo_values[start_cut:end_cut])

# Combine all cleaned segments
trigger_values_cleaned = np.concatenate(trigger_values_cleaned, axis=1)
sensor_values_cleaned = np.concatenate(sensor_values_cleaned, axis=0)
piezo_values_cleaned = np.concatenate(piezo_values_cleaned, axis=0)

# Print the cleaned msg data
print(f"Cleaned trigger data shape: {trigger_values_cleaned.shape}")

# Optional: plot the cleaned data
plt.figure(figsize=(10, 6))
plt.plot(trigger_values_cleaned.T)
plt.title('Cleaned Message')
plt.show()

print(f"Cleaned sensor values shape: {sensor_values_cleaned.shape}")
print(f"Cleaned piezo values shape: {piezo_values_cleaned.shape}")

labels = np.full(trigger_values_cleaned.shape[1], 0, dtype=int)
num_unlabeled = 0

# Assign labels based on which row in trigger_values_cleaned is active
for i in range(trigger_values_cleaned.shape[1]):  # Iterate over time steps
    active_indices = np.where(trigger_values_cleaned[:, i] == 1)[0]  # Find active trigger(s)
    if len(active_indices) > 0:
        labels[i] = active_indices[0]  # Assign the first active trigger index
    else:
        labels[i] = labels[i-1]  # Assign the previous label
        num_unlabeled += 1

print(f"Number of unlabeled time steps: {num_unlabeled}")
# Optional: Plot the labels

# Assuming 'labels' and 'trigger_values_cleaned.T[-1]' are already defined

plt.figure(figsize=(10, 3))

# Plot the 'labels' data
plt.plot(labels, marker='o', linestyle='None', markersize=3, label="Labels")

# Plot the 'trigger_values_cleaned.T[-1]' on the same axis
plt.plot(trigger_values_cleaned[-1], marker='x', linestyle='-', markersize=3, label="Trigger Values")

# Add title, labels, and legend
plt.title("Labels and Trigger Values over Time")
plt.xlabel("Time Step")
plt.ylabel("Trigger Index")
plt.legend()
plt.grid()
# Show the plot
plt.show()

#save the labels
np.save(labels_path, labels)
#save the cleaned data
saved_path_sensor = os.path.join(script_dir,"data\\records_final", test, folder1, 'sensor_values_cleaned.npy')
saved_path_piezo = os.path.join(script_dir,"data\\records_final", test, folder1, f'{finger}_cleaned.npy')
np.save(saved_path_sensor, sensor_values_cleaned)
np.save(saved_path_piezo, piezo_values_cleaned)