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

test_n = 4

folder=f"{test_type}\{test_type}_{test_n}"


labels=[]
labels2=[]

# Define the relative path to the file

sensor_values_path = os.path.join(script_dir, "data", macro_folder, test, folder, 'sensor_values_downsampled.npy')
labels_path = os.path.join(script_dir, "data", macro_folder, test, folder, 'labels.npy')   #pressure levels
labels_path2 = os.path.join(script_dir, "data", macro_folder, test, folder, 'labels2.npy') #pressure/no pressure
# Load the data
sensor_values = abs(np.load(sensor_values_path))  # Force sensor data
plt.plot(sensor_values[:,2])
#plt.show()
for i in range(len(sensor_values[:,2])):
    if sensor_values[i][2] < 0.14:
        labels2.append(0)
    else:
        labels2.append(1)
    if sensor_values[i][2] < 1:
        labels.append(0)
    elif sensor_values[i][2] < 3:
        labels.append(1)
    elif sensor_values[i][2] < 5:
        labels.append(2)
    elif sensor_values[i][2] < 7:
        labels.append(3)
    else:
        labels.append(4)
labels = np.array(labels)
print("len sensor values = ", len(sensor_values))
print("len labels = ", len(labels))
plt.plot(labels)
plt.plot(sensor_values[:,2])
#plt.show()
np.save(labels_path, labels)
np.save(labels_path2, labels2)