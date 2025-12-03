%% -------------------------------------------------------------
% FIR per finestre costanti di label (uno per label * finestra * canale)
% Usa ARX con na = 0 (solo parte FIR)
% Richiede Neuropixel.readNPY sul path
% --------------------------------------------------------------

clear; clc; close all;

%% PARAMETRI DI BASE

script_dir   = pwd;
macro_folder = 'records_final';

% === CONFIGURAZIONE ESPERIMENTO ===
test      = 'thumb_pressure';   % es. 'thumb_pressure', 'little_level_4', ...
finger    = 'thumb';            % 'index', 'middle', 'ring', 'little', 'thumb'

% File prodotti dai tuoi concatenater_*:
%   labels_concatenated.npy   : [N x 1]
%   {finger}_concatenated.npy : [N x 8]
labels_file = 'labels_concatenated.npy';
X_file      = [finger '_concatenated.npy'];

base_path   = fullfile(script_dir, 'data', macro_folder, test);

labels_path = fullfile(base_path, labels_file);
X_path      = fullfile(base_path, X_file);

% === PARAMETRI PER LE FINESTRE ===
windowLength = 100;   % numero di campioni per finestra
overlap      = 0.99;  % es. 0.99 -> passo 1 campione
stepSize     = max(1, floor(windowLength * (1 - overlap)));

% === PARAMETRI DEL FIR (ARX) ===
fir_order = 10;       % nb
na        = 0;        % niente parte AR
nk        = 0;        % nessun ritardo (puoi metterlo a 1 se vuoi input delay)

%% CARICAMENTO DATI

fprintf('Carico dati da:\n%s\n%s\n', X_path, labels_path);

X = Neuropixel.readNPY(X_path);       % [N x 8]
labels = Neuropixel.readNPY(labels_path); 
labels = double(labels(:));           % [N x 1]

[numSamples, numChannels] = size(X);
fprintf('Samples: %d, Channels: %d\n', numSamples, numChannels);

if length(labels) ~= numSamples
    error('labels_concatenated e %s_concatenated hanno lunghezze diverse!', finger);
end

%% LABEL UNICHE

unique_labels = unique(labels);
numLabels     = numel(unique_labels);

fprintf('Label uniche trovate (%d): ', numLabels);
disp(unique_labels.')

%% STEP 1: FINITRATURA E FILTRAGGIO FINESTRE CON LABEL COSTANTE

% Per ogni label avremo un cell array di finestre:
% windows_per_label{l} è una cell: ogni elemento è [windowLength x numChannels]
windows_per_label = cell(numLabels, 1);
windows_labels    = cell(numLabels, 1);  % memorizza le label (costanti) per controllo

winCountTotal = 0;

fprintf('\nInizio windowing con windowLength = %d, overlap = %.2f (stepSize = %d)\n', ...
    windowLength, overlap, stepSize);

startIdx = 1;

while startIdx + windowLength - 1 <= numSamples
    idxRange = startIdx : (startIdx + windowLength - 1);
    
    label_win = labels(idxRange);
    
    % Verifico se le label nella finestra sono tutte uguali
    if all(label_win == label_win(1))
        lbl_val = label_win(1);
        
        % Trovo l'indice nel vettore delle label uniche
        lIdx = find(unique_labels == lbl_val);
        
        X_win = X(idxRange, :);   % [windowLength x numChannels]
        
        % Inizializza la cell array se vuota
        if isempty(windows_per_label{lIdx})
            windows_per_label{lIdx} = {X_win};
            windows_labels{lIdx}    = lbl_val;
        else
            windows_per_label{lIdx}{end+1} = X_win; %#ok<AGROW>
            windows_labels{lIdx}(end+1,1)  = lbl_val; %#ok<AGROW>
        end
        
        winCountTotal = winCountTotal + 1;
    end
    
    startIdx = startIdx + stepSize;
end

fprintf('Totale finestre con label costante: %d\n', winCountTotal);
for lIdx = 1:numLabels
    nW = 0;
    if ~isempty(windows_per_label{lIdx})
        nW = numel(windows_per_label{lIdx});
    end
    fprintf('  Label %d: %d finestre\n', unique_labels(lIdx), nW);
end

%% STEP 2 + 3: ADDDESTRAMENTO FIR PER OGNI FINESTRA, PER OGNI CANALE

% Struttura:
% firModels{lIdx, ch}{wIdx} = modello FIR (ARX) per:
%  - label unique_labels(lIdx)
%  - canale ch
%  - finestra wIdx

firModels = cell(numLabels, numChannels);

fprintf('\nInizio training FIR per ogni label / canale / finestra...\n');

for lIdx = 1:numLabels
    lbl_val = unique_labels(lIdx);
    
    winCell = windows_per_label{lIdx};
    if isempty(winCell)
        fprintf('  [ATTENZIONE] Nessuna finestra per label %d, salto.\n', lbl_val);
        continue;
    end
    
    nWins = numel(winCell);
    fprintf('\nLabel %d: %d finestre da addestrare\n', lbl_val, nWins);
    
    for wIdx = 1:nWins
        X_win = winCell{wIdx};  % [windowLength x numChannels]
        
        % y: qui usi la LABEL come output costante (come hai richiesto)
        % se in futuro vuoi usare un sensore (es. forza), sostituisci qui.
        y_win = lbl_val * ones(windowLength, 1);
        
        for ch = 1:numChannels
            u_win = X_win(:, ch);
            
            data_id    = iddata(y_win, u_win, 1);     % Ts=1
            modelOrders = [na fir_order nk];         % [0 nb nk]
            
            model = arx(data_id, modelOrders);
            
            % Memorizzo modello
            if isempty(firModels{lIdx, ch})
                firModels{lIdx, ch} = {model};
            else
                firModels{lIdx, ch}{end+1} = model; %#ok<AGROW>
            end
        end
    end
end

fprintf('\nTraining FIR completato.\n');

%% SALVATAGGIO MODELLI

savePath = fullfile(base_path, ...
    sprintf('FIR_models_%s_win%d_ov%.2f_firord%d.mat', ...
    finger, windowLength, overlap, fir_order));

% save(savePath, 'firModels', 'unique_labels', ...
%     'windowLength', 'overlap', 'fir_order', 'na', 'nk');

fprintf('Modelli FIR salvati in:\n  %s\n', savePath);
