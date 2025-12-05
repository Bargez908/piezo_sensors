#load C:\Users\Davide\Documents\piezo_sensors\data\records_final\thumb_pressure\FIR_mean_matrix.npy and print it
import numpy as np
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'data', 'records_final', 'thumb_pressure', 'FIR_mean_matrix.npy')
data = np.load(file_path)
print(data.shape)
#print the fir values of the first channel of the for each label
for label_idx in range(data.shape[0]):
    print(f"Label {label_idx+1}, Channel 1 FIR coefficients:")
    print(data[label_idx, 0])
#print the sum of absolute values of the fir coefficients of the first channel for each label
for label_idx in range(data.shape[0]):
    sum_abs = np.sum(np.abs(data[label_idx, 0]))
    print(f"Label {label_idx+1}, Channel 1 sum of absolute FIR coefficients: {sum_abs}")