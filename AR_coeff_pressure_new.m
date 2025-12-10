%% -------------------------------------------------------------
% GENERATORE AR COEFFS PER MULTIPLI PARAMETRI
% Usa arburg()
% Salva file nel formato: order_windowLength_overlap_arCoeffs.npy
%                         order_windowLength_overlap_noiseVariances.npy
% --------------------------------------------------------------

clear; clc; close all;

%% === PATH & DATASET CONFIG ===
script_dir   = pwd;
macro_folder = 'records_final';

test      = 'thumb_pressure';
finger    = 'thumb';

fileName = [finger '_concatenated.npy'];
piezo_path = fullfile(script_dir, 'data', macro_folder, test, fileName);

fprintf("Caricamento dati: %s\n", piezo_path);
data = Neuropixel.readNPY(piezo_path);   % [N × numChannels]

[numSamples, numChannels] = size(data);

%% === PARAMETRI MULTI-RUN ===

orders         = [5, 10, 15];           % ordini AR da testare
windowLengths  = [80, 135, 313];        % campioni per finestra
overlaps       = [0.50, 0.80, 0.97];    % percentuali (0.50 = 50%)

save_dir = fullfile(script_dir, 'data', macro_folder, test, 'ar_coeff');

%% === LOOP SU TUTTE LE COMBINAZIONI ===

for order = orders
for windowLength = windowLengths
for overlap = overlaps
    
    fprintf("\n---------------------------------------------\n");
    fprintf("Running AR extraction: order=%d  win=%d  overlap=%.2f \n", ...
        order, windowLength, overlap);
    fprintf("---------------------------------------------\n");

    % Compute step size
    stepSize = max(1, floor(windowLength * (1 - overlap)));

    % Compute number of windows
    numWindows = floor((numSamples - windowLength) / stepSize) + 1;

    % Allocate arrays
    arCoeffs       = zeros(numChannels, order+1, numWindows);
    noiseVariances = zeros(numChannels, numWindows);

    %% === AR extraction ===
    for ch = 1:numChannels
        channelData = data(:, ch);

        wIdx = 1;
        for startIdx = 1:stepSize:(numSamples - windowLength + 1)
            windowData = channelData(startIdx : startIdx + windowLength - 1);

            % AR Burg
            [coeffs, noiseVar] = arburg(windowData, order);

            arCoeffs(ch, :, wIdx)       = coeffs;
            noiseVariances(ch, wIdx)    = noiseVar;
            wIdx = wIdx + 1;
        end
    end

    %% === COSTRUZIONE NOMI FILE ===

    % overlap in percentuale
    overlap_percent = round(overlap * 100);

    base_name = sprintf('%d_%d_%d', order, windowLength, overlap_percent);

    save_AR_path    = fullfile(save_dir, [base_name '_arCoeffs.npy']);
    save_Noise_path = fullfile(save_dir, [base_name '_noiseVariances.npy']);

    %% === SALVATAGGIO ===
    Neuropixel.writeNPY(arCoeffs,       save_AR_path);
    Neuropixel.writeNPY(noiseVariances, save_Noise_path);

    fprintf("SALVATO: %s\n", save_AR_path);
    fprintf("SALVATO: %s\n", save_Noise_path);

end
end
end

fprintf("\n=== FINITO TUTTI I PARAMETRI AR ===\n");
