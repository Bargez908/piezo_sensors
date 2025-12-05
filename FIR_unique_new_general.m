%% -------------------------------------------------------------
% FIR per finestre costanti (uno per label × finestra × canale)
% Usa ARX (na = 0)
% Richiede Neuropixel.readNPY sul path
% --------------------------------------------------------------

clear; clc; close all;

%% PARAMETRI DI BASE

script_dir   = pwd;
macro_folder = 'records_final';

do_train = false;
do_test  = true;

label_to_test = 2;      % quale label testare
n_seq_windows = 10;     % quante finestre sequenziali usare


% === CONFIGURAZIONE ESPERIMENTO ===
test      = 'thumb_pressure';     % es: 'thumb_pressure', 'little_level_4'
finger    = 'thumb';              % es: 'index','middle','ring','little','thumb'

% === CLASSIFICATION o REGRESSION ===
type_fir = "regression";          % "classification" o "regression"

% === FILE concatenati prodotti dal tuo concatenater ===
labels_file = 'labels_concatenated.npy';
X_file      = [finger '_concatenated.npy'];

base_path   = fullfile(script_dir, 'data', macro_folder, test);

labels_path = fullfile(base_path, labels_file);
X_path      = fullfile(base_path, X_file);

% Y per regressione
if strcmp(type_fir, "regression")
    Y_path = fullfile(base_path, 'sensor_values_concatenated.npy');
    Y      = Neuropixel.readNPY(Y_path);    % es [N × 3]
    Y      = Y(:,3);                         % prendi Fz
    figure;
    plot(Y, 'LineWidth', 1.5);
    grid on;
    xlabel('Sample Index');
    ylabel('Force (Z)');
    title(sprintf('Forza Z concatenata (%s)', test));

end

%% PARAMETRI FINESTRE
windowLength = 200;    % numero di campioni
overlap      = 0.97;  
stepSize     = max(1, floor(windowLength * (1 - overlap)));

%% PARAMETRI DEL FIR
fir_order = 10;        % nb
na        = 0;         % niente AR
nk        = 0;         % nessun ritardo

%% --- CARICAMENTO DATI ---

fprintf('Carico dati da:\n%s\n%s\n', X_path, labels_path);

X = Neuropixel.readNPY(X_path);        % [N x 8]
labels = Neuropixel.readNPY(labels_path);
labels = double(labels(:));            % [N x 1]

[numSamples, numChannels] = size(X);

fprintf('Samples: %d, Channels: %d\n', numSamples, numChannels);

if length(labels) ~= numSamples
    error('labels_concatenated e %s_concatenated hanno lunghezze diverse!', finger);
end

%% ================================================================
% TRAIN/TEST SPLIT (sequenziale, NO shuffle)
% ================================================================

train_ratio = 0.8;   % <-- QUI decidi quanto usare per training (es. 0.8 = 80%)

train_end = floor(numSamples * train_ratio);

fprintf('\nSuddivisione dati: %.0f%% train, %.0f%% test\n', ...
        train_ratio*100, (1-train_ratio)*100);

% TRAIN
X_train      = X(1:train_end, :);
labels_train = labels(1:train_end);

if strcmp(type_fir, "regression")
    Y_train = Y(1:train_end);
end

% TEST (lo userai dopo, NON nelle FIR)
X_test      = X(train_end+1:end, :);
labels_test = labels(train_end+1:end);

if strcmp(type_fir, "regression")
    Y_test = Y(train_end+1:end);
end

fprintf("TRAIN samples: %d\n", size(X_train,1));
fprintf("TEST  samples: %d\n", size(X_test,1));

% DA QUI IN POI, USA SOLO IL TRAIN SET!!
X     = X_train;
labels = labels_train;
if strcmp(type_fir, "regression")
    Y = Y_train;
end

[numSamples, numChannels] = size(X);


%% --- LABEL UNICHE ---

unique_labels = unique(labels);
numLabels     = numel(unique_labels);

fprintf('Label uniche trovate (%d): ', numLabels);
disp(unique_labels.')

%% ================================================================
% STEP 1: WINDOWING + SCARTO FINESTRE NON COSTANTI
%  → salviamo X_win + idxRange originale
% ================================================================

function [windows_per_label, idxRanges_per_label, winCountTotal] = ...
         window_and_filter(X, labels, unique_labels, windowLength, stepSize)

    numLabels = numel(unique_labels);
    [numSamples, ~] = size(X);

    windows_per_label   = cell(numLabels, 1);
    idxRanges_per_label = cell(numLabels, 1);
    winCountTotal = 0;

    startIdx = 1;

    while startIdx + windowLength - 1 <= numSamples

        idxRange = startIdx : (startIdx + windowLength - 1);
        label_win = labels(idxRange);

        % Finestra valida solo se tutte le label sono uguali
        if all(label_win == label_win(1))
            lbl_val = label_win(1);
            lIdx    = find(unique_labels == lbl_val);

            X_win = X(idxRange, :);

            if isempty(windows_per_label{lIdx})
                windows_per_label{lIdx}   = {X_win};
                idxRanges_per_label{lIdx} = {idxRange};
            else
                windows_per_label{lIdx}{end+1}   = X_win;
                idxRanges_per_label{lIdx}{end+1} = idxRange;
            end

            winCountTotal = winCountTotal + 1;
        end

        startIdx = startIdx + stepSize;
    end
end

% WINDOWING TRAIN
[windows_per_label, idxRanges_per_label, train_total] = ...
    window_and_filter(X_train, labels_train, unique_labels, windowLength, stepSize);

fprintf("Finestre TRAIN valide: %d\n", train_total);


% WINDOWING TEST
[windows_per_label_test, idxRanges_per_label_test, test_total] = ...
    window_and_filter(X_test, labels_test, unique_labels, windowLength, stepSize);

fprintf("Finestre TEST valide: %d\n", test_total);

for lIdx = 1:numLabels
    nW = numel(windows_per_label{lIdx});
    fprintf('  Label %d → %d finestre\n', unique_labels(lIdx), nW);
end

%% ================================================================
% STEP 2 + 3: TRAINING FIR PER OGNI (label, finestra, canale)
% ================================================================
if do_train
    % Risultato:
    % firModels{labelIndex, channel}{windowIndex} = modello FIR
    firModels = cell(numLabels, numChannels);

    fprintf('\nTraining FIR per ogni label / finestra / canale...\n');

    for lIdx = 1:numLabels
        lbl_val = unique_labels(lIdx);
        winCell = windows_per_label{lIdx};
        idxCell = idxRanges_per_label{lIdx};
        
        if isempty(winCell)
            fprintf('[ATTENZIONE] Nessuna finestra per label %d\n', lbl_val);
            continue;
        end
        
        nWins = numel(winCell);
        fprintf('\nLabel %d: %d finestre\n', lbl_val, nWins);
        
        for wIdx = 1:nWins
            
            X_win = winCell{wIdx};                % [windowLength × 8]
            idxRange_local = idxCell{wIdx};       % indici originali
            
            % =======================
            % COSTRUZIONE y_win
            % =======================
            if strcmp(type_fir, "classification")
                y_win = lbl_val * ones(windowLength,1);
            elseif strcmp(type_fir, "regression")
                y_win = Y(idxRange_local);
            else
                error("type_fir deve essere 'classification' o 'regression'");
            end
            
            % =======================
            % TRAINING PER OGNI CANALE
            % =======================
            for ch = 1:numChannels
                u_win = X_win(:, ch);
                
                data_id     = iddata(y_win, u_win, 1);        % Ts=1
                modelOrders = [na fir_order nk];             % [0 nb nk]
                
                model = arx(data_id, modelOrders);           % stima FIR
                
                % Salva nella struttura finale
                firModels{lIdx, ch}{wIdx} = model;
            end
            %fprintf('  Finestra %d/%d completata.\n', wIdx, nWins);
        end
        fprintf('Training per label %d completato.\n', lbl_val);
    end

    fprintf('\nTraining FIR completato.\n');

    %% ================================================================
    % CALCOLO MEDIA E VARIANZA DEI COEFFICIENTI FIR PER LABEL E CANALE
    % ================================================================

    fir_mean = cell(numLabels, numChannels);
    fir_var  = cell(numLabels, numChannels);

    fprintf('\nCalcolo media e varianza dei coefficienti FIR...\n');

    for lIdx = 1:numLabels
        lbl_val = unique_labels(lIdx);

        for ch = 1:numChannels

            models_cell = firModels{lIdx, ch};

            if isempty(models_cell)
                fprintf('[Label %d, Channel %d] Nessun modello FIR.\n', lbl_val, ch);
                continue;
            end
            
            % Estrai tutti i coefficienti FIR B
            coeff_matrix = [];

            for wIdx = 1:numel(models_cell)
                model = models_cell{wIdx};
                B = model.B;                     % vettore FIR [1 × (fir_order+1)]
                coeff_matrix = [coeff_matrix; B];  %#ok<AGROW>
            end

            % MEDIA e VARIANZA
            fir_mean{lIdx, ch} = mean(coeff_matrix, 1);      % [1 × nb+1]
            fir_var{lIdx, ch}  = var(coeff_matrix, 0, 1);     % [1 × nb+1]

            % -----------------------------------------
            % STAMPA VALORI REALI (media e varianza)
            % -----------------------------------------
            fprintf("\n==============================\n");
            fprintf("LABEL %d  |  CANALE %d\n", lbl_val, ch);
            fprintf("==============================\n");

            fprintf("Media coeff FIR:\n");
            disp(fir_mean{lIdx, ch});

            fprintf("Varianza coeff FIR:\n");
            disp(fir_var{lIdx, ch});
            fprintf("--------------------------------\n");
        end
    end

    fprintf('\nMedia e varianza FIR calcolate e stampate.\n');


    %% ================================================================
    % SALVATAGGIO
    % ================================================================

    savePath = fullfile(base_path, ...
        sprintf('FIR_models_%s_win%d_ov%.2f_firord%d.mat', ...
        finger, windowLength, overlap, fir_order));

    save(savePath, 'firModels', 'unique_labels', ...
        'windowLength', 'overlap', 'fir_order', 'na', 'nk', '-v7.3');

    fprintf('\nDimensioni modelli FIR nella prima finestra:\n');

    nb_plus1 = length(fir_mean{1,1});  % numero coeff FIR per ogni finestra

    FIR_mean_matrix = zeros(numLabels, numChannels, nb_plus1);
    FIR_var_matrix  = zeros(numLabels, numChannels, nb_plus1);

    for lIdx = 1:numLabels
        for ch = 1:numChannels
            FIR_mean_matrix(lIdx, ch, :) = fir_mean{lIdx, ch};
            FIR_var_matrix(lIdx,  ch, :) = fir_var{lIdx, ch};
        end
    end

    Neuropixel.writeNPY(FIR_mean_matrix, fullfile(base_path, 'FIR_mean_matrix.npy'));
    Neuropixel.writeNPY(FIR_var_matrix,  fullfile(base_path, 'FIR_var_matrix.npy'));
    fprintf('FIR mean e varianza salvati come matrici NPY.\n');


    for lIdx = 1:numLabels
        for ch = 1:numChannels
            if ~isempty(firModels{lIdx,ch})
                m = firModels{lIdx,ch}{1};
                fprintf("Label %d, Channel %d, FIR size(B) = %s\n", ...
                    unique_labels(lIdx), ch, mat2str(size(m.B)));
            end
        end
    end


    fprintf('\nModelli FIR salvati in:\n  %s\n', savePath);
end

%% ================================================================
% TEST FIR: applica tutti i FIR medi a n finestre di una label
% ================================================================
if do_test
    FIR_mean_matrix = Neuropixel.readNPY(fullfile(base_path,'FIR_mean_matrix.npy'));
    FIR_var_matrix  = Neuropixel.readNPY(fullfile(base_path,'FIR_var_matrix.npy'));
    fprintf("Caricati FIR_mean_matrix e FIR_var_matrix da disco.\n");

    fprintf('\n========== FASE DI TEST FIR ==========\n');

    % Trova indice della label da testare
    idx_label_test = find(unique_labels == label_to_test);
    if isempty(idx_label_test)
        error("La label_to_test = %d non esiste nelle unique_labels!", label_to_test);
    end

    test_windows = windows_per_label_test{idx_label_test};

    if isempty(test_windows)
        error("Nessuna finestra di TEST disponibile per la label %d", label_to_test);
    end

    if length(test_windows) < n_seq_windows
        error("Solo %d finestre di TEST per label %d, ma ne hai richieste %d", ...
            length(test_windows), label_to_test, n_seq_windows);
    end

    % Prendi le prime n finestre sequenziali
    X_test_seq = test_windows(1:n_seq_windows);

    fprintf('Testing su label %d con %d finestre di test.\n', ...
        label_to_test, n_seq_windows);

    % Loop su tutti i FIR medi (label × canale)
    for lIdx = 1:numLabels
        for ch = 1:numChannels

            % Coefficienti FIR medi per questa (label, canale)
            B = squeeze(FIR_mean_matrix(lIdx, ch, :));   % [nb+1 x 1]

            y_means = zeros(n_seq_windows, 1);

            % Applica il FIR a ciascuna finestra di test
            for w = 1:n_seq_windows
                X_win = X_test_seq{w};          % [windowLength x numChannels]
                u     = X_win(:, ch);           % canale ch

                % Usiamo filter (FIR causale)
                y_hat = filter(B, 1, u);        % stessa lunghezza di u

                % Salviamo la media della risposta su quella finestra
                y_means(w) = mean(y_hat);
            end

            fprintf('\n--- FIR[label=%d, ch=%d] ---\n', unique_labels(lIdx), ch);

            for w = 1:n_seq_windows
                fprintf('  Finestra %d → media(y_hat) = %.6f\n', w, y_means(w));
            end
        end
    end

    fprintf('========== FINE TEST FIR ==========\n');

end

