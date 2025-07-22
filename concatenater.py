import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# Creazione di un semplice grafico
script_dir = os.path.dirname(os.path.abspath(__file__))

macro_folder = "records_final"
test = "little_pressure"
test_type = "pressure"  # "sliding" or "pressure"
finger="little"  # "index", "middle", "ring", "little", "thumb"

macro_folder = "records_final"
test = "little_level_4"
test_type = "sliding"  # "sliding" or "pressure"
finger="little"  # "index", "middle", "ring", "little", "thumb"

n_concatenations = 5
sensor_values_path = []
labels_path = []
labels_path2 = []
piezo_data_path = []
#cycle with i going up until there is no file with that name
for i in range(n_concatenations):
    folder = f"{test_type}\{test_type}_{i}"
    #if i ==2 skip this cycle
    # if i == 2:
    #     continue
    sensor_values_path.append(os.path.join(script_dir,  "data", macro_folder, test, folder, 'sensor_values_downsampled.npy'))
    labels_path.append(os.path.join(script_dir,         "data", macro_folder, test, folder, 'labels.npy'))   #pressure levels
    labels_path2.append(os.path.join(script_dir,        "data", macro_folder, test, folder, 'labels2.npy')) #pressure/no pressure
    piezo_data_path.append(os.path.join(script_dir,     "data", macro_folder, test, folder, f'{finger}_downsampled.npy'))
#concatenate the data checking the results have the same length
sensor_values = []
labels = []
labels2 = []
piezo_values = []
for i in range(n_concatenations): 
    sensor_values.append(np.load(sensor_values_path[i]))
    labels.append(np.load(labels_path[i]))
    labels2.append(np.load(labels_path2[i]))
    piezo_values.append(np.load(piezo_data_path[i]))

sensor_values = np.concatenate(sensor_values, axis=0)
labels = np.concatenate(labels, axis=0)
labels2 = np.concatenate(labels2, axis=0)
piezo_values = np.concatenate(piezo_values, axis=0)
print("len sensor values = ", len(sensor_values))
print("len labels = ", len(labels))
print("len labels2 = ", len(labels2))
print("len piezo values = ", len(piezo_values))
# try if all the arrays have the same length else raise an error
if not (len(sensor_values) == len(labels) == len(labels2) == len(piezo_values)):
    raise ValueError("The arrays have different lengths")
# Save the concatenated data
concatenated_sensor_values_path = os.path.join(script_dir, "data", macro_folder, test, 'sensor_values_concatenated.npy')
concatenated_labels_path = os.path.join(script_dir, "data", macro_folder, test, 'labels_concatenated.npy')
concatenated_labels2_path = os.path.join(script_dir, "data", macro_folder, test, 'labels2_concatenated.npy')
concatenated_piezo_values_path = os.path.join(script_dir, "data", macro_folder, test, f'{finger}_concatenated.npy')

np.save(concatenated_sensor_values_path, sensor_values)
np.save(concatenated_labels_path, labels)
np.save(concatenated_labels2_path, labels2)
np.save(concatenated_piezo_values_path, piezo_values)
# Plot the concatenated data of the sensor values and labes
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(sensor_values[:, 2], label='Sensor Values')
plt.title('Concatenated Sensor Values')
plt.xlabel('Sample Index')
plt.ylabel('Sensor Value')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(labels, label='Labels', color='orange')
plt.plot(labels2, label='Labels2', color='green')
plt.title('Concatenated Labels')
plt.xlabel('Sample Index')
plt.ylabel('Label Value')
plt.legend()
plt.tight_layout()
plt.show()