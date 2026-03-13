# FIR

## Training

### File da lanciare (training FIR)

1. `def_fir_classification_multithread.m`

Input principali:
- `data/records_final/<finger>_<test>/<test>/<test>_0/<finger>_downsampled.npy`
- `data/records_final/<finger>_<test>/<test>/<test>_0/labels.npy`
- `data/records_final/<finger>_<test>/<test>/<test>_1/<finger>_downsampled.npy`
- `data/records_final/<finger>_<test>/<test>/<test>_1/labels.npy`
- `data/records_final/<finger>_<test>/<test>/<test>_2/<finger>_downsampled.npy`
- `data/records_final/<finger>_<test>/<test>/<test>_2/labels.npy`
- `data/records_final/<finger>_<test>/<test>/<test>_3/<finger>_downsampled.npy`
- `data/records_final/<finger>_<test>/<test>/<test>_3/labels.npy`
- `data/records_final/<finger>_<test>/<test>/<test>_4/<finger>_downsampled.npy`
- `data/records_final/<finger>_<test>/<test>/<test>_4/labels.npy`

Output principali (se `save_model = true`):
- `data/records_final/<finger>_<test>/fir_models/FIR_MODEL_<wl>_<ov>_<fir_order>_<fold_mask>.mat`

Output in workspace MATLAB:
- `results(combo_idx).FIR_mean_matrix`
- `results(combo_idx).windows_per_fold`

Note operative:
- Per addestrare davvero: `train_fir = true`
- Per salvare i modelli `.mat`: `save_model = true`

---

## Pipeline file caricati -> output FIR

Questa e la pipeline dati tipica per arrivare ai modelli FIR.

1. **Downsampling e allineamento (upstream)**
   - Script: `downsampler_new.py`
   - Legge:
     - `piezo_<finger>.npy`
     - `sensor_data.npy`
     - `received_trigger.npy`
   - Fa:
     - taglio iniziale temporale
     - interpolazione su asse tempo piezo
   - Salva:
     - `<finger>_downsampled.npy`
     - `sensor_values_downsampled.npy`
     - `triggers_downsampled.npy`

2. **Creazione label pressione (upstream)**
   - Script: `labeler pressure.py`
   - Legge:
     - `sensor_values_downsampled.npy`
   - Fa:
     - threshold sul canale forza (`sensor_values[:,2]`)
   - Salva:
     - `labels.npy` (classi 0..4)
     - `labels2.npy` (binaria)

3. **Training FIR**
   - Script: `def_fir_classification_multithread.m`
   - Legge:
     - `<finger>_downsampled.npy`
     - `labels.npy`
   - Salva:
     - `FIR_MODEL_*.mat`

---

## Passi training FIR (esaustivo)

Di seguito i passaggi che esegue `def_fir_classification_multithread.m` quando `train_fir = true`.

1. **Config base**
   - Definisce:
     - `test` (es. `pressure` o `sliding`)
     - `finger` (es. `thumb`)
     - `n_folds = 5`
     - griglia iperparametri:
       - `window_lengths`
       - `overlaps`
       - `fir_orders`
   - Definisce anche:
     - `na = 0`, `nk = 0` (FIR puro in ARX)
     - `label_offset = 1`
     - flag `train_fir`, `test_fir`, `save_model`

2. **Costruzione path dei 5 dataset**
   - Costruisce:
     - `<dataset_root>/<test>_0`
     - ...
     - `<dataset_root>/<test>_4`
   - Esempio con `finger=thumb`, `test=pressure`:
     - `data/records_final/thumb_pressure/pressure/pressure_0`
     - ...
     - `data/records_final/thumb_pressure/pressure/pressure_4`

3. **Caricamento segnali e label**
   - Per ogni dataset:
     - legge `X = <finger>_downsampled.npy`
     - legge `labels = labels.npy`
   - Controlla che:
     - `length(labels) == size(X,1)`
   - Estrae l elenco globale delle label uniche (`unique_labels`).

4. **Loop su tutte le combinazioni della griglia**
   - Per ogni `wl` in `window_lengths`
   - Per ogni `ov` in `overlaps`
   - Per ogni `fir_order` in `fir_orders`
   - Calcola `stepSize = max(1, floor(wl*(1-ov)))`.

5. **Windowing per ciascuno dei 5 dataset**
   - Fa windowing sull intero segnale di ogni dataset, non separando prima per label.
   - Ogni finestra viene analizzata:
     - se tutte le label nella finestra sono uguali, la finestra e valida
     - se la finestra attraversa una transizione label, viene scartata
   - Le finestre valide sono poi organizzate in:
     - `windows_per_label{l}`
     - `idxRanges_per_label{l}`

6. **Equalizzazione dataset/label**
   - Conta quante finestre valide ci sono per ogni coppia `(dataset, label)`.
   - Prende il minimo globale:
     - `global_min = min(counts_before(:))`
   - Per ogni dataset e label:
     - se ha piu di `global_min` finestre, tiene solo le prime `global_min`
     - quindi il taglio avviene dalla coda (ultime finestre eliminate)
   - Obiettivo:
     - training bilanciato tra classi e tra dataset.

7. **5-fold CV leave-one-dataset-out**
   - In ogni fold:
     - 1 dataset e test
     - gli altri 4 dataset sono train
   - Si ripete 5 volte, cosi ogni dataset viene testato una volta.

8. **Aggregazione finestre di train (per label)**
   - Per una label `l`, concatena le finestre di `l` provenienti dai 4 dataset di train.
   - Questa lista unificata e `winCellAll`.
   - `nWins = numel(winCellAll)` e il numero reale di esempi train per quella label nel fold.

9. **Training FIR con ARX (per canale, in parallelo)**
   - Per ogni canale (`parfor ch = 1:numChannels`) e per ogni finestra:
     - `u_win = X_win(:, ch)` (ingresso)
     - `y_win = (label + label_offset) * ones(wl,1)` (target costante)
     - crea `iddata(y_win, u_win, 1)`
     - stima `model = arx(data_id, [na fir_order nk])`
   - Con `na=0, nk=0` il modello e FIR puro; `fir_order` e il numero di coefficienti `B`.

10. **Media coefficienti FIR**
    - Per ogni `(fold, label, canale)`:
      - raccoglie i vettori `B` ottenuti da tutte le finestre
      - fa media riga per riga:
        - `mean(coeff_matrix, 1)`
    - Risultato:
      - un FIR medio per quella label/canale/fold.
    - Viene salvato in:
      - `FIR_mean_matrix(fold, lIdx, ch, :)`

11. **Salvataggio modelli**
    - Se `save_model = true`, salva un `.mat` per fold:
      - `FIR_MODEL_<wl>_<ov>_<fir_order>_<fold_mask>.mat`
    - Contenuto tipico:
      - `FIR_mean_matrix_fold`
      - `unique_labels`
      - metadati (`wl`, `ov`, `fir_order`, `fold`, `fold_mask`, `label_offset`, `na`, `nk`).

12. **Risultato finale training**
    - Per ogni tripla di iperparametri ottieni 5 modelli (uno per fold).
    - Numero totale modelli:
      - `len(window_lengths) * len(overlaps) * len(fir_orders) * n_folds`
    - Con valori attuali:
      - `3 * 2 * 3 * 5 = 90` modelli.

---

## Test

### File da lanciare (test FIR)

1. `def_fir_classification_multithread.m`

Input principali:
- Modelli FIR salvati:
  - `data/records_final/<finger>_<test>/fir_models/FIR_MODEL_<wl>_<ov>_<fir_order>_<fold_mask>.mat`
- Dataset (gli stessi 5 del training):
  - `data/records_final/<finger>_<test>/<test>/<test>_0/<finger>_downsampled.npy`
  - `data/records_final/<finger>_<test>/<test>/<test>_0/labels.npy`
  - `...`
  - `data/records_final/<finger>_<test>/<test>/<test>_4/<finger>_downsampled.npy`
  - `data/records_final/<finger>_<test>/<test>/<test>_4/labels.npy`

Output principali:
- Per ogni modello:
  - `<MODEL>_predictions_pairwise.csv`
  - `<MODEL>_pairwise_error_stats.csv`
  - `<MODEL>_equalization_counts.csv`
  - `<MODEL>_plot.png`
  - `<MODEL>_pairwise_var_cm.png`
- Riassunto globale:
  - `all_models_pairwise_error_stats.csv`
- CM medie fold:
  - `FIR_MODEL_<wl>_<ov>_<fir_order>_11111_pairwise_var_cm.png`

Cartella output test:
- `data/records_final/<finger>_<test>/csv_results/fir_test_results/`

Note operative:
- Per testare: `test_fir = true`
- Se usi modelli gia salvati: `train_fir = false`
- Il test usa i `.mat` presenti in `fir_models`

---

## Pipeline test (file caricati -> output)

1. **Caricamento dataset base**
   - Lo script carica comunque i 5 dataset (`X`, `labels`) in memoria.

2. **Scansione modelli**
   - Cerca tutti i file `FIR_MODEL_*.mat` in `fir_models`.

3. **Per ogni modello**
   - Legge metadati (`wl`, `ov`, `fir_order`, `fold/test_ds`, `label_offset`).
   - Rifà windowing + filtro finestre miste su tutti i 5 dataset.
   - Equalizza nel test con minimo globale `(dataset,label)`.
   - Valuta il solo dataset di test del fold, confrontando ogni `true_label` contro tutti i FIR di label (`fir_label`).
   - Salva CSV/plot/CM del modello.

4. **Aggregazione finale**
   - Concatena tutte le tabelle pairwise in `all_models_pairwise_error_stats.csv`.
   - Aggiunge righe aggregate `11111` (media sui 5 fold) per ogni combinazione `(wl, ov, fir_order, true_label, fir_label)`.
   - Salva anche le CM aggregate `11111`.

---

## Passi test FIR (esaustivo)

Di seguito i passaggi che esegue `def_fir_classification_multithread.m` quando `test_fir = true`.

1. **Controllo cartelle**
   - Verifica che esista `fir_models`.
   - Crea (se manca) la cartella output test:
     - `data/records_final/<finger>_<test>/csv_results/fir_test_results/`

2. **Lettura lista modelli**
   - Cerca tutti i `.mat` con pattern `FIR_MODEL_*.mat`.
   - Se non trova modelli, produce tabella vuota e termina.

3. **Load e validazione modello**
   - Carica il `.mat`.
   - Verifica campi minimi:
     - `FIR_mean_matrix_fold`
     - `unique_labels`
   - Se mancanti, salta il modello.

4. **Parsing metadati**
   - Recupera:
     - `windowLength`
     - `overlap`
     - `fir_order`
     - `test_ds` (fold)
     - `label_offset`
   - Se necessario, fa fallback parsando il nome file.

5. **Windowing test su tutti i dataset**
   - Con i parametri del modello (`wl`,`ov`) rifà:
     - windowing completo
     - scarto finestre mixed-label
   - Lo fa su tutti i 5 dataset.

6. **Equalizzazione test**
   - Calcola `counts_before` per ogni `(dataset,label)`.
   - Prende `global_min`.
   - Tronca tutte le celle a `global_min`.
   - Salva anche `counts_after`.

7. **Selezione dataset di test del fold**
   - Usa solo `test_ds` per la valutazione del modello corrente.
   - Mantiene i risultati equalizzati.

8. **Valutazione pairwise completa**
   - Per ogni `true_label` nel test set:
     - prende tutte le sue finestre.
   - Per ogni `fir_label`:
     - usa i coefficienti FIR di quella label.
     - produce `predicted_label` per finestra.
     - calcola `prediction_error = predicted_label - true_label_shifted`.

9. **Statistiche pairwise**
   - Per ogni coppia `(true_label, fir_label)` calcola:
     - `error_mean`
     - `error_variance`
     - `n_windows`

10. **Salvataggi per modello**
    - `predictions_pairwise.csv` (dettaglio per finestra)
    - `pairwise_error_stats.csv` (tabella 5x5 o NxN)
    - `equalization_counts.csv` (before/after equalizzazione)
    - `plot.png` (true vs predicted su diagonale FIR)
    - `pairwise_var_cm.png` (CM varianza stile heatmap)

11. **Summary globale**
    - Concatena tutte le `pairwise_error_stats` dei modelli fold nel file:
      - `all_models_pairwise_error_stats.csv`

12. **Media fold `11111`**
    - Per ogni combinazione `(wl, ov, fir_order, true_label, fir_label)`:
      - media su 5 fold di:
        - `error_mean`
        - `error_variance`
        - (anche campi numerici di supporto come media)
    - Aggiunge righe con:
      - `model_file = FIR_MODEL_<wl>_<ov>_<fir_order>_11111.mat`
      - `test_dataset = 0`
    - Salva CM media `11111` per ogni combinazione `(wl,ov,fir_order)`.

---

## Link dati

https://liveunibo-my.sharepoint.com/:f:/g/personal/davide_bargellini2_unibo_it/Et1w4u5-hZ5MmiJ7NF8gM9QBGvveZhVsIYJxcQovzrqT6Q?e=B4O5ct
