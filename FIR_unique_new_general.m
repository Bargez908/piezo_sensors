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

label_to_test = 4;      % quale label testare
n_seq_windows = 1;     % quante finestre sequenziali usare

plot_y = false;        % se true, plotta le finestre Y usate nel training
maxWinsToPlot = 15;  % quante finestre per label vuoi plottare

% === CONFIGURAZIONE ESPERIMENTO ===
test      = 'thumb_pressure';     % es: 'thumb_pressure', 'little_level_4'
finger    = 'thumb';              % es: 'index','middle','ring','little','thumb'

% === CLASSIFICATION o REGRESSION ===
type_fir = "regression";          % "classification" o "regression"

train_ratio = 0.8;   % (es. 0.8 = 80%)

%% PARAMETRI FINESTRE
windowLength = 200;    % numero di campioni
overlap      = 0.50;  
stepSize     = max(1, floor(windowLength * (1 - overlap)));

%% PARAMETRI DEL FIR
fir_order = 10;        % nb
na        = 0;         % niente AR
nk        = 0;         % nessun ritardo

%% --- CARICAMENTO DATI ---

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
    Y      = abs(Y);                       % prendi valore assoluto
    % figure;
    % plot(Y, 'LineWidth', 1.5);
    % grid on;
    % xlabel('Sample Index');
    % ylabel('Force (Z)');
    % title(sprintf('Forza Z concatenata (%s)', test));

end

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

%% ------------------------------------------------
% PLOT: prime N finestre di Y usate nel TRAINING
% ------------------------------------------------

if plot_y
    fprintf("\nVisualizzazione finestre Y del TRAIN prima del training FIR...\n");

    for lIdx = 1:numLabels
        lbl_val = unique_labels(lIdx);
        idxCell = idxRanges_per_label{lIdx};

        if isempty(idxCell)
            fprintf('[Plot] Nessuna finestra TRAIN per label %d, salto.\n', lbl_val);
            continue;
        end

        nWins = numel(idxCell);
        nToPlot = min(maxWinsToPlot, nWins);

        figure('Name', sprintf('TRAIN - Label %d - prime %d finestre Y', ...
                               lbl_val, nToPlot), ...
               'NumberTitle', 'off');

        for w = 1:nToPlot
            idxRange = idxCell{w};
            y_win = Y(idxRange);   % Y_train in quella finestra

            subplot(nToPlot, 1, w);
            plot(y_win, 'LineWidth', 1.0);
            grid on;
            ylabel(sprintf('W%d', w));

            if w == 1
                title(sprintf('TRAIN - Label %d: prime %d finestre Y', ...
                              lbl_val, nToPlot));
            end
        end

        xlabel('Sample dentro la finestra');
    end
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

    % save(savePath, 'firModels', 'unique_labels', ...
    %     'windowLength', 'overlap', 'fir_order', 'na', 'nk', '-v7.3');

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

    %% ================================================================
    % PASSO A: STATISTICA DEGLI OUTPUT FIR (score mean e score std)
    % ================================================================

    fprintf("\nCalcolo statistica degli output FIR (score mean + score std)...\n");

    FIR_score_mean = zeros(numLabels, numChannels);
    FIR_score_std  = zeros(numLabels, numChannels);

    for lIdx = 1:numLabels
        lbl_val = unique_labels(lIdx);

        winCell = windows_per_label{lIdx};  % finestre TRAIN di questa label

        if isempty(winCell)
            fprintf('[ATTENZIONE] Nessuna finestra TRAIN per label %d\n', lbl_val);
            continue;
        end

        for ch = 1:numChannels

            B = squeeze(FIR_mean_matrix(lIdx, ch, :)).';   % vettore FIR [1 × nb+1]

            scores = zeros(1, numel(winCell));

            for w = 1:numel(winCell)
                X_win = winCell{w};
                u     = X_win(:, ch);

                y_hat = filter(B, 1, u);  
                scores(w) = mean(y_hat);
            end

            FIR_score_mean(lIdx, ch) = mean(scores);
            FIR_score_std(lIdx, ch)  = std(scores);
        end
    end

    fprintf("Statistica FIR calcolata.\n");

    % Salva
    Neuropixel.writeNPY(FIR_score_mean, fullfile(base_path,"FIR_score_mean.npy"));
    Neuropixel.writeNPY(FIR_score_std,  fullfile(base_path,"FIR_score_std.npy"));

    fprintf("Salvati FIR_score_mean e FIR_score_std in NPY.\n");


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

    % Matrice: [finestra × canale × label]
    y_store = zeros(n_seq_windows, numChannels, numLabels);

    % Loop su tutti i FIR medi (label × canale)
    for lIdx = 1:numLabels
        for ch = 1:numChannels

            % Coefficienti FIR medi per questa (label, canale)
            B = squeeze(FIR_mean_matrix(lIdx, ch, :));   % [nb+1 x 1]

            % Applica il FIR a ciascuna finestra di test
            for w = 1:n_seq_windows
                X_win = X_test_seq{w};          
                u     = X_win(:, ch);           

                y_hat = filter(B, 1, u);
                y_val = mean(y_hat);

                y_store(w, ch, lIdx) = y_val;
            end
        end
    end

    %% ===== STAMPA RISULTATI =====

    for lIdx = 1:numLabels
        fprintf("\n===== RISULTATI PER FIR DELLA LABEL %d =====\n", unique_labels(lIdx));
        
        for ch = 1:numChannels
            fprintf("\n--- FIR[label=%d, ch=%d] ---\n", unique_labels(lIdx), ch);

            for w = 1:n_seq_windows
                fprintf("  Finestra %d → media(y_hat) = %.6f\n", w, y_store(w,ch,lIdx));
            end
        end

        % === MEDIA DEI CANALI PER OGNI FINESTRA ===
        fprintf("\n>> MEDIA TRA I CANALI (per ogni finestra):\n");
        for w = 1:n_seq_windows
            mean_channels = mean(y_store(w,:,lIdx));
            fprintf("  Finestra %d → media_canali = %.6f\n", w, mean_channels);
        end

        % === MEDIA TOTALE (canali + finestre) ===
        fprintf("\n>> MEDIA TOTALE LABEL %d = %.6f\n", ...
            unique_labels(lIdx), mean(y_store(:,:,lIdx), "all"));
    end
    %% ================================================================
    % STAMPA FINALE: MEDIA TOTALE PER LABEL (dopo tutto il test)
    % ================================================================

    fprintf("\n\n==============================\n");
    fprintf("   MEDIA TOTALE PER LABEL\n");
    fprintf("==============================\n");

    for lIdx = 1:numLabels
        global_mean = mean(y_store(:,:,lIdx), "all");
        fprintf("Label %d → media totale = %.6f\n", unique_labels(lIdx), global_mean);
    end
    fprintf('========== FINE TEST FIR ==========\n');
    %% ================================================================
    % PASSO B: CLASSIFICAZIONE ONLINE BASATA SU FIR
    % ================================================================
    
    fprintf("\n\n========== CLASSIFICAZIONE ONLINE FIR ==========\n");

    % Carica statistiche FIR (media, std)
    FIR_score_mean = Neuropixel.readNPY(fullfile(base_path,"FIR_score_mean.npy"));
    FIR_score_std  = Neuropixel.readNPY(fullfile(base_path,"FIR_score_std.npy"));

    % Preallocazione vettore errori per ciascuna label
    Z_label = zeros(numLabels, 1);

    % Consideriamo SOLO la prima finestra (o usa un loop sulle finestre)
    w_test = 1;

    fprintf("\nClassificazione usando la finestra di test %d\n", w_test);

    for lIdx = 1:numLabels
        
        z_accum = 0;

        for ch = 1:numChannels

            % score dinamico ottenuto dal FIR (lo hai già in y_store)
            y_val = y_store(w_test, ch, lIdx);
            
            % media e std del FIR sui dati di training
            mu  = FIR_score_mean(lIdx, ch);
            sig = FIR_score_std(lIdx, ch) + 1e-6;  % evita divisioni per zero
            
            % z-score normalizzato e accumulato
            z_accum = z_accum + abs((y_val - mu) / sig);
        end
        
        Z_label(lIdx) = z_accum;  % errore totale per questa label
    end

    % Label predetta = la label con Z più basso
    [~, best_lIdx] = min(Z_label);
    predicted_label = unique_labels(best_lIdx);

    fprintf("\n===== RISULTATO CLASSIFICAZIONE FIR =====\n");
    fprintf("Label scelta per il TEST : %d\n", label_to_test);
    fprintf("Label predetta dal FIR   : %d\n\n", predicted_label);

    fprintf("Z per ogni label:\n");
    for lIdx = 1:numLabels
        fprintf("  Label %d → Z = %.4f\n", unique_labels(lIdx), Z_label(lIdx));
    end
    fprintf("==========================================\n\n");


end

