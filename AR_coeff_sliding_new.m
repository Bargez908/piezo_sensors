%% -------------------------------------------------------------
% GENERATORE AR COEFFS PER SLIDING DATASET SINGOLO
% Usa arburg()
% Input:  data/records_final/<finger>_level_<level>/sliding/sliding_<test_n>/<finger>_cleaned.npy
% Output: .../ar_coeff/order_windowLength_overlap_arCoeffs.npy
%         .../ar_coeff/order_windowLength_overlap_noiseVariances.npy
% --------------------------------------------------------------

clear; clc; close all;

%% === PATH & DATASET CONFIG ===
script_dir   = pwd;
macro_folder = 'records_final';

finger    = 'thumb';
level     = 2;
test_n    = 5;
test_type = 'sliding';

test = sprintf('%s_level_%d', finger, level);
fileName = [finger '_cleaned.npy'];

dataset_dir = fullfile(script_dir, 'data', macro_folder, test, test_type, ...
    sprintf('%s_%d', test_type, test_n));
piezo_path = fullfile(dataset_dir, fileName);

fprintf("Caricamento dati: %s\n", piezo_path);

if ~exist(piezo_path, 'file')
    error('Input file not found: %s', piezo_path);
end

data = Neuropixel.readNPY(piezo_path);   % [N x numChannels]
[numSamples, numChannels] = size(data);

fprintf("Samples: %d | Channels: %d\n", numSamples, numChannels);

%% === PARAMETRI MULTI-RUN ===
orders        = [5, 10, 15];
windowLengths = [25, 100, 400];
overlaps      = [0.50, 1];

save_dir = fullfile(dataset_dir, 'ar_coeff');

%% === LOOP SU TUTTE LE COMBINAZIONI ===
for order = orders
for windowLength = windowLengths
for overlap = overlaps

    fprintf("\n---------------------------------------------\n");
    fprintf("Running AR extraction: order=%d  win=%d  overlap=%.2f \n", ...
        order, windowLength, overlap);
    fprintf("---------------------------------------------\n");

    stepSize = max(1, floor(windowLength * (1 - overlap)));

    if numSamples < windowLength
        warning('Skipping order=%d win=%d overlap=%.2f: numSamples (%d) < windowLength (%d)', ...
            order, windowLength, overlap, numSamples, windowLength);
        continue;
    end

    numWindows = floor((numSamples - windowLength) / stepSize) + 1;

    arCoeffs       = zeros(numChannels, order + 1, numWindows);
    noiseVariances = zeros(numChannels, numWindows);

    for ch = 1:numChannels
        channelData = data(:, ch);

        wIdx = 1;
        for startIdx = 1:stepSize:(numSamples - windowLength + 1)
            windowData = channelData(startIdx : startIdx + windowLength - 1);

            [coeffs, noiseVar] = arburg(windowData, order);

            arCoeffs(ch, :, wIdx)    = coeffs;
            noiseVariances(ch, wIdx) = noiseVar;
            wIdx = wIdx + 1;
        end
    end

    overlap_percent = round(overlap * 100);

    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end

    base_name = sprintf('%d_%d_%d', order, windowLength, overlap_percent);

    save_AR_path    = fullfile(save_dir, [base_name '_arCoeffs.npy']);
    save_Noise_path = fullfile(save_dir, [base_name '_noiseVariances.npy']);

    Neuropixel.writeNPY(arCoeffs,       save_AR_path);
    Neuropixel.writeNPY(noiseVariances, save_Noise_path);

    fprintf("SALVATO: %s\n", save_AR_path);
    fprintf("SALVATO: %s\n", save_Noise_path);

end
end
end

fprintf("\n=== FINITO TUTTI I PARAMETRI AR SLIDING ===\n");
