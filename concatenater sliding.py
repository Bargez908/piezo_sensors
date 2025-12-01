import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))

###############################################################################
# CONFIGURAZIONE
###############################################################################
macro_folder = "records_final"

finger = "thumb"    # "index", "middle", "ring", "little", "thumb"
level  = 3          # 2,3,4,... in base alla tua cartella
n_concatenations = 5 # quanti sliding_x usare (0,1,2,3,4,...)

###############################################################################
# COSTRUZIONE DEL PATH BASE
###############################################################################
# Esempio risultato:
# data/records_final/little_level_4/sliding/sliding_0/
#
test_folder = f"{finger}_level_{level}"

base_path = os.path.join(
    script_dir, "data", macro_folder, test_folder, "sliding"
)

###############################################################################
# COSTRUZIONE LISTE DEI PATH sliding_0, sliding_1, ...
###############################################################################
sensor_values_paths = []
labels_paths        = []
piezo_paths         = []

for i in range(n_concatenations):

    sliding_folder = os.path.join(base_path, f"sliding_{i}")

    sensor_values_paths.append(
        os.path.join(sliding_folder, "sensor_values_cleaned.npy")
    )
    labels_paths.append(
        os.path.join(sliding_folder, "labels.npy")
    )
    piezo_paths.append(
        os.path.join(sliding_folder, f"{finger}_cleaned.npy")
    )

###############################################################################
# CARICAMENTO E CONCATENAZIONE
###############################################################################
sensor_values_list = [np.load(p) for p in sensor_values_paths]
labels_list        = [np.load(p) for p in labels_paths]
piezo_list         = [np.load(p) for p in piezo_paths]

sensor_values = np.concatenate(sensor_values_list, axis=0)
labels        = np.concatenate(labels_list, axis=0)
piezo_values  = np.concatenate(piezo_list, axis=0)

print("len sensor values =", len(sensor_values))
print("len labels       =", len(labels))
print("len piezo values =", len(piezo_values))

###############################################################################
# CHECK
###############################################################################
if not (len(sensor_values) == len(labels) == len(piezo_values)):
    raise ValueError("ERRORE: sensor_values, labels e piezo_values NON hanno la stessa lunghezza")

###############################################################################
# SALVATAGGIO OUTPUT
###############################################################################
concat_dir = os.path.join(script_dir, "data", macro_folder, test_folder)

np.save(os.path.join(concat_dir, "sensor_values_concatenated.npy"), sensor_values)
np.save(os.path.join(concat_dir, "labels_concatenated.npy"),       labels)
np.save(os.path.join(concat_dir, f"{finger}_concatenated.npy"),    piezo_values)

print("SALVATO OK in:", concat_dir)
