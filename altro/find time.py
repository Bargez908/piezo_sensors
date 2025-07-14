import matplotlib.pyplot as plt
import numpy as np
import os

# Creazione di un semplice grafico
script_dir = os.path.dirname(os.path.abspath(__file__))
finger="index_2"

# Define the relative path to the file
sensor_data_path = os.path.join(script_dir, 'pressure', finger, 'sensor_values_downsampled.npy')
labels_path = os.path.join(script_dir, 'pressure', finger, 'labels')

values = np.load(sensor_data_path)

idle_points = []
rise_points = []
steady_points = []
fall_points = []
old = 0
plt.title("Seleziona i punti con il mouse e premi Invio per terminare")
plt.plot(values[:,2])
plt.grid()

# Utilizzo di ginput per selezionare punti
print("Clicca sui punti desiderati e premi Invio per terminare")
selected_points = plt.ginput(n=-1, timeout=0)


# Visualizzazione dei punti selezionati
selected_points = np.array(selected_points)
x= selected_points[:,0]
#x in int
x = x.astype(int)
for i in range(len(x)+1):
    if i == len(x):
        for j in range (old, len(values)): 
            idle_points.append(j)
        break
    if i%4 == 0:
        for j in range (old, x[i]):
            idle_points.append(j)
    if i%4 == 1:
        for j in range (old, x[i]):
            rise_points.append(j)
    if i%4 == 2:
        for j in range (old, x[i]):
            steady_points.append(j)
    if i%4 == 3:
        for j in range (old, x[i]):
            fall_points.append(j)
    old = x[i]


#plot the points
plt.plot(idle_points, values[idle_points,2], 'ro')
plt.plot(rise_points, values[rise_points,2], 'go')
plt.plot(steady_points, values[steady_points,2], 'bo')
plt.plot(fall_points, values[fall_points,2], 'yo')
print("initial lenght points: ", len(values[:,2]))
print("total lenght points: ", len(idle_points)+len(rise_points)+len(steady_points)+len(fall_points))
#plot the values array also
plt.plot(values[:,2])
#save the points in C:/Users/utente/Documents/signals/pressure/index_1/labels
if not os.path.exists(labels_path):
    os.makedirs(labels_path)
np.save(labels_path + '/idle_points.npy', idle_points)
np.save(labels_path + '/rise_points.npy', rise_points)
np.save(labels_path + '/steady_points.npy', steady_points)
np.save(labels_path + '/fall_points.npy', fall_points)

# Mostra il grafico
plt.show()
