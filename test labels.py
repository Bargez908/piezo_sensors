#load C:\Users\utente\Documents\signals\pressure\index_1\labels\idle_points.npy
import numpy as np
import matplotlib.pyplot as plt
import os

#load the points
idle_points = np.load('C:/Users/utente/Documents/piezosensor_classification/data/ar10/pressure_1/sensor_values_downsampled.npy')
idle_points_2 = np.load('C:/Users/utente/Documents/piezosensor_classification/data/ar10/pressure_1/piezo_index.npy')
# rise_points = np.load('C:/Users/utente/Documents/signals/pressure/index_1/labels/rise_points.npy')
# steady_points = np.load('C:/Users/utente/Documents/signals/pressure/index_1/labels/steady_points.npy')
# fall_points = np.load('C:/Users/utente/Documents/signals/pressure/index_1/labels/fall_points.npy')
# values = np.load('C:/Users/utente/Documents/signals/pressure/index_1/sensor_values_downsampled.npy')

#plot the points

plt.plot(idle_points_2)
#grid on
plt.title('Idle points')
plt.xlabel('Time')  
#activate grid
plt.grid(True)
plt.show()
