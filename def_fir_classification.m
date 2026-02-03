%% -------------------------------------------------------------
% FIR classification with 5-fold CV over separate datasets
% - Loads 5 separate datasets (e.g., pressure_0 ... pressure_4)
% - Windowing + discard mixed-label windows
% - Equalize labels across datasets by keeping first windows (drop from end)
% - Train FIR per (label, channel) on each fold and average coefficients
% --------------------------------------------------------------

clear; clc; close all;

%% BASE CONFIG
script_dir   = pwd;
macro_folder = 'records_final';

% === EXPERIMENT CONFIG ===
test   = 'pressure';   % e.g. 'pressure', 'sliding'
finger = 'thumb';      % 'index', 'middle', 'ring', 'little', 'thumb'

% 5 datasets: {test}_0 ... {test}_4
n_folds = 5;
dataset_indices = 0:(n_folds-1);
dataset_prefix  = test;   % folder name prefix (pressure_0, pressure_1, ...)
dataset_root    = fullfile(script_dir, 'data', macro_folder, [finger '_' test], test);

X_file      = [finger '_downsampled.npy'];  % [N x 8]
labels_file = 'labels.npy';                 % [N x 1]

% === WINDOW PARAMETERS (grid) ===
window_lengths = [200];
overlaps       = [0.50];

% === FIR (ARX) PARAMETERS (grid) ===
fir_orders  = [10]; % nb
na          = 0;    % no AR
nk          = 0;    % no delay
label_offset = 1;   % shift labels by +1 to avoid y=0

% === EQUALIZATION ===
do_equalize = true;

% === SAVE MODELS ===
save_model = false;

%% -------------------------------------------------------------
% LOAD DATASETS
%% -------------------------------------------------------------

dataset_paths = cell(n_folds, 1);
for i = 1:n_folds
    ds_idx = dataset_indices(i);
    dataset_paths{i} = fullfile(dataset_root, sprintf('%s_%d', dataset_prefix, ds_idx));
end

datasets = struct('X', [], 'labels', [], ...
                  'windows_per_label', [], 'idxRanges_per_label', []);

all_labels = [];

fprintf('=== DATASET LOADING ===\n');
for d = 1:n_folds
    X_path = fullfile(dataset_paths{d}, X_file);
    L_path = fullfile(dataset_paths{d}, labels_file);

    fprintf('\nDataset %d: %s\n', d, dataset_paths{d});
    fprintf('  X: %s\n  L: %s\n', X_path, L_path);

    X = Neuropixel.readNPY(X_path);
    labels = Neuropixel.readNPY(L_path);
    labels = double(labels(:));

    [numSamples, numChannels] = size(X);
    fprintf('  Samples: %d | Channels: %d\n', numSamples, numChannels);

    if length(labels) ~= numSamples
        error('Length mismatch in dataset %d: labels vs X', d);
    end

    datasets(d).X = X;
    datasets(d).labels = labels;
    all_labels = [all_labels; labels];
end

unique_labels = unique(all_labels);
numLabels     = numel(unique_labels);

fprintf('\nUnique labels found (%d): ', numLabels);
disp(unique_labels.');

%% -------------------------------------------------------------
% GRID SEARCH: windowLength x overlap x fir_order
%% -------------------------------------------------------------

[~, numChannels] = size(datasets(1).X);

results = struct('windowLength', {}, 'overlap', {}, 'fir_order', {}, ...
                 'FIR_mean_matrix', {}, 'windows_per_fold', {});

if save_model
    save_root = fullfile(script_dir, 'data', macro_folder, [finger '_' test], 'fir_models');
    if ~exist(save_root, 'dir')
        mkdir(save_root);
    end
end

combo_idx = 0;

for wl = window_lengths
    for ov = overlaps
        stepSize = max(1, floor(wl * (1 - ov)));

        for fir_order = fir_orders
            combo_idx = combo_idx + 1;

            fprintf('\n========================================\n');
            fprintf('GRID COMBO %d | wl=%d, ov=%.2f, fir_order=%d\n', ...
                combo_idx, wl, ov, fir_order);
            fprintf('========================================\n');

            %% WINDOWING + FILTER MIXED LABELS (per dataset)
            fprintf('\n=== WINDOWING (filter mixed labels) ===\n');
            for d = 1:n_folds
                [windows_per_label, idxRanges_per_label, winCountTotal] = ...
                    window_and_filter(datasets(d).X, datasets(d).labels, unique_labels, wl, stepSize);

                datasets(d).windows_per_label   = windows_per_label;
                datasets(d).idxRanges_per_label = idxRanges_per_label;

                fprintf('Dataset %d: total valid windows = %d\n', d, winCountTotal);
                print_label_counts(windows_per_label, unique_labels, '  ');
            end

            %% EQUALIZE LABELS ACROSS DATASETS (keep first windows, drop from end)
            if do_equalize
                fprintf('\n=== EQUALIZATION (keep first windows per label) ===\n');

                counts_before = count_windows_per_label(datasets, unique_labels);
                fprintf('Counts BEFORE equalization:\n');
                print_counts_table(counts_before, unique_labels);

                global_min = min(counts_before(:));
                fprintf('Global minimum across all labels/datasets: %d\n', global_min);

                for d = 1:n_folds
                    for lIdx = 1:numLabels
                        nW = numel(datasets(d).windows_per_label{lIdx});
                        if nW > global_min
                            datasets(d).windows_per_label{lIdx}   = datasets(d).windows_per_label{lIdx}(1:global_min);
                            datasets(d).idxRanges_per_label{lIdx} = datasets(d).idxRanges_per_label{lIdx}(1:global_min);
                        end
                    end
                end

                counts_after = count_windows_per_label(datasets, unique_labels);
                fprintf('\nCounts AFTER equalization:\n');
                print_counts_table(counts_after, unique_labels);
            end

            %% 5-FOLD TRAINING (leave-one-dataset-out)
            nb_coeff = fir_order;  % arx nb = number of B coefficients

            FIR_mean_matrix = zeros(n_folds, numLabels, numChannels, nb_coeff);
            windows_per_fold = zeros(n_folds, numLabels);

            fprintf('\n=== 5-FOLD TRAINING ===\n');

            for fold = 1:n_folds
                test_ds  = fold;
                train_ds = setdiff(1:n_folds, test_ds);

                fprintf('\n[FOLD %d] Test dataset: %d | Train datasets: %s\n', ...
                    fold, test_ds, mat2str(train_ds));

                for lIdx = 1:numLabels
                    lbl_val = unique_labels(lIdx);
                    winCellAll = {};

                    for d = train_ds
                        if ~isempty(datasets(d).windows_per_label{lIdx})
                            winCellAll = [winCellAll, datasets(d).windows_per_label{lIdx}]; %#ok<AGROW>
                        end
                    end

                    nWins = numel(winCellAll);
                    windows_per_fold(fold, lIdx) = nWins;

                    if nWins == 0
                        fprintf('  [WARN] Fold %d: no windows for label %d\n', fold, lbl_val);
                        continue;
                    end

                    fprintf('  Label %d: %d windows\n', lbl_val, nWins);

                    % Train FIR per channel, then average coefficients across windows
                    for ch = 1:numChannels
                        coeff_matrix = zeros(nWins, nb_coeff);

                        for wIdx = 1:nWins
                            X_win = winCellAll{wIdx};        % [wl x numChannels]
                            y_win = (lbl_val + label_offset) * ones(wl, 1);
                            u_win = X_win(:, ch);

                            data_id    = iddata(y_win, u_win, 1);
                            modelOrders = [na fir_order nk];
                            model = arx(data_id, modelOrders);

                            B = model.B;
                            if numel(B) ~= nb_coeff
                                error('B length mismatch: expected %d, got %d', nb_coeff, numel(B));
                            end
                            coeff_matrix(wIdx, :) = B;
                        end

                        FIR_mean_matrix(fold, lIdx, ch, :) = mean(coeff_matrix, 1);
                    end
                end
            end

            fprintf('\nTraining completed for wl=%d, ov=%.2f, fir_order=%d.\n', ...
                wl, ov, fir_order);

            if save_model
                for fold = 1:n_folds
                    fold_mask = repmat('0', 1, n_folds);
                    fold_mask(fold) = '1';

                    FIR_mean_matrix_fold = squeeze(FIR_mean_matrix(fold, :, :, :));

                    filename = sprintf('FIR_MODEL_%d_%.2f_%d_%s.mat', ...
                        wl, ov, fir_order, fold_mask);
                    save_path = fullfile(save_root, filename);

                    save(save_path, 'FIR_mean_matrix_fold', 'unique_labels', ...
                        'wl', 'ov', 'fir_order', 'fold', 'fold_mask', 'label_offset', ...
                        'windowLength', 'overlap', 'na', 'nk');
                end
            end

            % Store results in workspace
            results(combo_idx).windowLength    = wl;
            results(combo_idx).overlap         = ov;
            results(combo_idx).fir_order       = fir_order;
            results(combo_idx).FIR_mean_matrix = FIR_mean_matrix;
            results(combo_idx).windows_per_fold = windows_per_fold;
        end
    end
end

fprintf('\nAll grid combinations completed.\n');

%% -------------------------------------------------------------
% LOCAL FUNCTIONS
%% -------------------------------------------------------------

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

        if all(label_win == label_win(1))
            lbl_val = label_win(1);
            lIdx = find(unique_labels == lbl_val);

            X_win = X(idxRange, :);

            if isempty(windows_per_label{lIdx})
                windows_per_label{lIdx}   = {X_win};
                idxRanges_per_label{lIdx} = {idxRange};
            else
                windows_per_label{lIdx}{end+1}   = X_win; %#ok<AGROW>
                idxRanges_per_label{lIdx}{end+1} = idxRange; %#ok<AGROW>
            end

            winCountTotal = winCountTotal + 1;
        end

        startIdx = startIdx + stepSize;
    end
end

function print_label_counts(windows_per_label, unique_labels, indent)
    for lIdx = 1:numel(unique_labels)
        nW = numel(windows_per_label{lIdx});
        fprintf('%sLabel %d: %d windows\n', indent, unique_labels(lIdx), nW);
    end
end

function counts = count_windows_per_label(datasets, unique_labels)
    n_folds = numel(datasets);
    numLabels = numel(unique_labels);
    counts = zeros(n_folds, numLabels);
    for d = 1:n_folds
        for lIdx = 1:numLabels
            counts(d, lIdx) = numel(datasets(d).windows_per_label{lIdx});
        end
    end
end

function print_counts_table(counts, unique_labels)
    [n_folds, numLabels] = size(counts);
    header = sprintf('      ');
    for lIdx = 1:numLabels
        header = [header, sprintf('L%d   ', unique_labels(lIdx))]; %#ok<AGROW>
    end
    fprintf('%s\n', header);
    for d = 1:n_folds
        row = sprintf('DS%02d: ', d);
        for lIdx = 1:numLabels
            row = [row, sprintf('%-4d', counts(d, lIdx))]; %#ok<AGROW>
        end
        fprintf('%s\n', row);
    end
end
