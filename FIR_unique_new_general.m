%% -------------------------------------------------------------
% FIR per finestre costanti (uno per label × finestra × canale)
% Usa ARX (na = 0)
% Richiede Neuropixel.readNPY sul path
% --------------------------------------------------------------

clear; clc; close all;

%% PARAMETRI DI BASE

script_dir   = pwd;
macro_folder = 'records_final';

% === CONFIGURAZIONE ESPERIMENTO ===
test      = 'thumb_pressure';     % es: 'thumb_pressure', 'little_level_4'
finger    = 'thumb';              % es: 'index','middle','ring','little','thumb'

% === CLASSIFICATION o REGRESSION ===
type_fir = "classification";          % "classification" o "regression"

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
windowLength = 313;    % numero di campioni
overlap      = 0.97;   % 99% → step = 1 campione
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

%% --- LABEL UNICHE ---

unique_labels = unique(labels);
numLabels     = numel(unique_labels);

fprintf('Label uniche trovate (%d): ', numLabels);
disp(unique_labels.')

%% ================================================================
% STEP 1: WINDOWING + SCARTO FINESTRE NON COSTANTI
%  → salviamo X_win + idxRange originale
% ================================================================

windows_per_label    = cell(numLabels, 1);
idxRanges_per_label  = cell(numLabels, 1);
winCountTotal = 0;

fprintf('\nWindowing: windowLength = %d, overlap = %.2f (stepSize = %d)\n', ...
    windowLength, overlap, stepSize);

startIdx = 1;

while startIdx + windowLength - 1 <= numSamples
    
    idxRange = startIdx : (startIdx + windowLength - 1);
    label_win = labels(idxRange);
    
    % Finestra valida solo se tutte le label sono uguali
    if all(label_win == label_win(1))
        
        lbl_val = label_win(1);
        lIdx    = find(unique_labels == lbl_val);
        
        X_win = X(idxRange, :);   % [windowLength × 8]
        
        % Salvo la finestra X
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

fprintf('Totale finestre valide: %d\n', winCountTotal);
for lIdx = 1:numLabels
    nW = numel(windows_per_label{lIdx});
    fprintf('  Label %d → %d finestre\n', unique_labels(lIdx), nW);
end

%% ================================================================
% STEP 2 + 3: TRAINING FIR PER OGNI (label, finestra, canale)
% ================================================================

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
        fprintf('  Finestra %d/%d completata.\n', wIdx, nWins);
    end
    fprintf('Training per label %d completato.\n', lbl_val);
end

fprintf('\nTraining FIR completato.\n');

%% ================================================================
% SALVATAGGIO
% ================================================================

savePath = fullfile(base_path, ...
    sprintf('FIR_models_%s_win%d_ov%.2f_firord%d.mat', ...
    finger, windowLength, overlap, fir_order));

%save(savePath, 'firModels', 'unique_labels', ...
%    'windowLength', 'overlap', 'fir_order', 'na', 'nk');

fprintf('\nModelli FIR salvati in:\n  %s\n', savePath);
