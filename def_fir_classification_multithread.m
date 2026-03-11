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
window_lengths = [25,100,400];
overlaps       = [0.50, 1];

% === FIR (ARX) PARAMETERS (grid) ===
fir_orders  = [5,10,15]; % nb
na          = 0;    % no AR
nk          = 0;    % no delay
label_offset = 1;   % shift labels by +1 to avoid y=0

% === EQUALIZATION ===
do_equalize = true;

% === EXECUTION FLAGS ===
train_fir = false;   % run ARX/FIR training
test_fir  = true;  % evaluate saved models in fir_models

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

save_root = fullfile(script_dir, 'data', macro_folder, [finger '_' test], 'fir_models');
if save_model
    if ~exist(save_root, 'dir')
        mkdir(save_root);
    end
end

% Start parallel pool only when training is requested
if train_fir && isempty(gcp('nocreate'))
    parpool('local');
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

            if train_fir
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

                        % Train FIR per channel in parallel, then average coefficients across windows
                        parfor ch = 1:numChannels
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

                        % Expose descriptive variable names for the .mat file
                        windowLength = wl; %#ok<NASGU>
                        overlap      = ov; %#ok<NASGU>

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
            else
                fprintf('\nTraining skipped for wl=%d, ov=%.2f, fir_order=%d (train_fir=false).\n', ...
                    wl, ov, fir_order);
            end
        end
    end
end

fprintf('\nAll grid combinations completed.\n');

if test_fir
    fprintf('\n=== FIR MODEL TESTING ===\n');

    if ~exist(save_root, 'dir')
        error('Model folder not found: %s', save_root);
    end

    test_output_root = fullfile(script_dir, 'data', macro_folder, [finger '_' test], 'csv_results', 'fir_test_results');
    if ~exist(test_output_root, 'dir')
        mkdir(test_output_root);
    end

    summary_table = test_saved_models(save_root, test_output_root, ...
        datasets, n_folds, na, nk, label_offset);

    summary_csv = fullfile(test_output_root, 'all_models_pairwise_error_stats.csv');
    writetable(summary_table, summary_csv);
    fprintf('Saved test summary CSV: %s\n', summary_csv);
end

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
    % Pretty print table of window counts with fixed column widths
    [n_folds, numLabels] = size(counts);

    colw = 10; % width per column for readability

    % Header
    header = '      ';
    for lIdx = 1:numLabels
        header = [header, sprintf(['%' num2str(colw) 's'], sprintf('L%d', unique_labels(lIdx)))]; %#ok<AGROW>
    end
    fprintf('%s\n', header);

    % Rows
    for d = 1:n_folds
        row = sprintf('DS%02d:', d);
        for lIdx = 1:numLabels
            row = [row, sprintf([' %' num2str(colw-1) 'd'], counts(d, lIdx))]; %#ok<AGROW>
        end
        fprintf('%s\n', row);
    end
end

function summary_table = test_saved_models(save_root, test_output_root, ...
        datasets, n_folds, na_default, nk_default, default_label_offset)

    model_files = dir(fullfile(save_root, 'FIR_MODEL_*.mat'));
    if isempty(model_files)
        warning('No FIR model files found in %s', save_root);
        summary_table = empty_variance_table();
        return;
    end

    summary_parts = cell(numel(model_files), 1);
    used_parts = 0;

    for m = 1:numel(model_files)
        model_name = model_files(m).name;
        model_path = fullfile(model_files(m).folder, model_name);

        fprintf('\n[TEST %d/%d] Loading model: %s\n', m, numel(model_files), model_name);
        try
            S = load(model_path);

            required_fields = {'FIR_mean_matrix_fold', 'unique_labels'};
            for f = 1:numel(required_fields)
                if ~isfield(S, required_fields{f})
                    warning('Skipping model %s: missing field %s', model_name, required_fields{f});
                    S = struct();
                    break;
                end
            end
            if isempty(fieldnames(S))
                continue;
            end

            [windowLength, overlap, fir_order, test_ds, model_label_offset] = ...
                parse_model_metadata(S, model_name, n_folds, default_label_offset);

            if test_ds < 1 || test_ds > n_folds
                warning('Skipping model %s: invalid test fold %d', model_name, test_ds);
                continue;
            end

            model_na = na_default;
            model_nk = nk_default;
            if isfield(S, 'na'), model_na = S.na; end
            if isfield(S, 'nk'), model_nk = S.nk; end

            if model_na ~= 0 || model_nk ~= 0
                warning('Model %s has na=%d nk=%d. Test uses FIR-only prediction from B coefficients.', ...
                    model_name, model_na, model_nk);
            end

            model_unique_labels = double(S.unique_labels(:));
            FIR_mean_matrix_fold = normalize_fir_matrix_shape(S.FIR_mean_matrix_fold, numel(model_unique_labels));

            [pred_table, var_table, equalization_table] = evaluate_model_on_testset(model_name, test_ds, ...
                datasets, FIR_mean_matrix_fold, model_unique_labels, ...
                windowLength, overlap, fir_order, model_label_offset);

            [~, model_base, ~] = fileparts(model_name);
            pred_csv = fullfile(test_output_root, [model_base '_predictions_pairwise.csv']);
            var_csv  = fullfile(test_output_root, [model_base '_pairwise_error_stats.csv']);
            eq_csv   = fullfile(test_output_root, [model_base '_equalization_counts.csv']);
            plot_png = fullfile(test_output_root, [model_base '_plot.png']);

            writetable(pred_table, pred_csv);
            writetable(var_table, var_csv);
            writetable(equalization_table, eq_csv);
            save_prediction_plot(pred_table, model_name, test_ds, windowLength, overlap, fir_order, plot_png);

            fprintf('  Saved prediction CSV: %s\n', pred_csv);
            fprintf('  Saved pairwise CSV:   %s\n', var_csv);
            fprintf('  Saved equalize CSV:   %s\n', eq_csv);
            fprintf('  Saved plot:           %s\n', plot_png);

            used_parts = used_parts + 1;
            summary_parts{used_parts} = var_table;
        catch ME
            warning('Skipping model %s: %s', model_name, ME.message);
            continue;
        end
    end

    if used_parts == 0
        summary_table = empty_variance_table();
    else
        summary_table = vertcat(summary_parts{1:used_parts});
    end
end

function [pred_table, var_table, equalization_table] = evaluate_model_on_testset(model_name, test_ds, ...
        datasets, FIR_mean_matrix_fold, model_unique_labels, ...
        windowLength, overlap, fir_order, model_label_offset)

    stepSize = max(1, floor(windowLength * (1 - overlap)));
    [windows_per_dataset, idxRanges_per_dataset, counts_before, counts_after, global_min] = ...
        window_filter_and_equalize_all_datasets(datasets, model_unique_labels, windowLength, stepSize);

    test_windows_per_label = windows_per_dataset{test_ds};
    test_idxRanges_per_label = idxRanges_per_dataset{test_ds};

    [model_numLabels, model_numChannels, ~] = size(FIR_mean_matrix_fold);
    [~, test_numChannels] = size(datasets(test_ds).X);

    if model_numLabels ~= numel(model_unique_labels)
        error('Model labels mismatch in %s', model_name);
    end
    if test_numChannels < model_numChannels
        error('Dataset channels (%d) < model channels (%d) for %s', ...
            test_numChannels, model_numChannels, model_name);
    end

    all_start_idx = [];
    all_true = [];
    all_true_raw = [];
    all_fir = [];
    all_fir_raw = [];
    all_pred = [];
    all_err = [];

    n_windows_by_pair = zeros(model_numLabels, model_numLabels);
    mean_error_by_pair = nan(model_numLabels, model_numLabels);
    var_error_by_pair = nan(model_numLabels, model_numLabels);

    for trueIdx = 1:model_numLabels
        true_lbl_val = model_unique_labels(trueIdx);
        winCell = test_windows_per_label{trueIdx};
        idxCell = test_idxRanges_per_label{trueIdx};
        nWins = numel(winCell);

        if nWins == 0
            continue;
        end

        start_vals = zeros(nWins, 1);
        for wIdx = 1:nWins
            start_vals(wIdx) = idxCell{wIdx}(1);
        end

        true_vals_raw = true_lbl_val * ones(nWins, 1);
        true_vals = (true_lbl_val + model_label_offset) * ones(nWins, 1);

        for firIdx = 1:model_numLabels
            fir_lbl_val = model_unique_labels(firIdx);
            fir_coeffs_label = squeeze(FIR_mean_matrix_fold(firIdx, :, :));

            pred_vals = zeros(nWins, 1);
            for wIdx = 1:nWins
                X_win = winCell{wIdx};
                pred_vals(wIdx) = predict_window_label(X_win, fir_coeffs_label);
            end

            err_vals = pred_vals - true_vals;
            fir_vals_raw = fir_lbl_val * ones(nWins, 1);
            fir_vals = (fir_lbl_val + model_label_offset) * ones(nWins, 1);

            all_start_idx = [all_start_idx; start_vals]; %#ok<AGROW>
            all_true = [all_true; true_vals]; %#ok<AGROW>
            all_true_raw = [all_true_raw; true_vals_raw]; %#ok<AGROW>
            all_fir = [all_fir; fir_vals]; %#ok<AGROW>
            all_fir_raw = [all_fir_raw; fir_vals_raw]; %#ok<AGROW>
            all_pred = [all_pred; pred_vals]; %#ok<AGROW>
            all_err = [all_err; err_vals]; %#ok<AGROW>

            n_windows_by_pair(trueIdx, firIdx) = nWins;
            mean_error_by_pair(trueIdx, firIdx) = mean(err_vals);
            var_error_by_pair(trueIdx, firIdx) = var(err_vals, 0);
        end
    end

    if isempty(all_start_idx)
        pred_table = empty_prediction_table();
    else
        sort_keys = [all_start_idx, all_true_raw, all_fir_raw];
        [~, order_idx] = sortrows(sort_keys, [1 2 3]);
        n_window = (1:numel(order_idx)).';

        n_rows = numel(order_idx);
        pred_table = table( ...
            repmat({model_name}, n_rows, 1), ...
            repmat(test_ds, n_rows, 1), ...
            repmat(windowLength, n_rows, 1), ...
            repmat(overlap, n_rows, 1), ...
            repmat(fir_order, n_rows, 1), ...
            n_window, ...
            all_start_idx(order_idx), ...
            all_true_raw(order_idx), ...
            all_true(order_idx), ...
            all_fir_raw(order_idx), ...
            all_fir(order_idx), ...
            all_pred(order_idx), ...
            all_err(order_idx), ...
            'VariableNames', {'model_file', 'test_dataset', 'window_length', ...
            'overlap', 'fir_order', 'n_window', 'start_idx', ...
            'true_label_raw', 'true_label', 'fir_label_raw', 'fir_label', ...
            'predicted_label', 'prediction_error'});
    end

    var_table = build_pairwise_error_table(model_name, test_ds, ...
        windowLength, overlap, fir_order, model_unique_labels, model_label_offset, ...
        n_windows_by_pair, mean_error_by_pair, var_error_by_pair, global_min);

    equalization_table = build_equalization_table(model_name, test_ds, ...
        windowLength, overlap, fir_order, model_unique_labels, model_label_offset, ...
        counts_before, counts_after, global_min);
end

function [windows_per_dataset, idxRanges_per_dataset, counts_before, counts_after, global_min] = ...
        window_filter_and_equalize_all_datasets(datasets, unique_labels, windowLength, stepSize)

    n_folds = numel(datasets);
    numLabels = numel(unique_labels);

    windows_per_dataset = cell(n_folds, 1);
    idxRanges_per_dataset = cell(n_folds, 1);
    counts_before = zeros(n_folds, numLabels);

    for d = 1:n_folds
        [windows_per_label, idxRanges_per_label, ~] = window_and_filter( ...
            datasets(d).X, datasets(d).labels, unique_labels, windowLength, stepSize);

        windows_per_dataset{d} = windows_per_label;
        idxRanges_per_dataset{d} = idxRanges_per_label;

        for lIdx = 1:numLabels
            counts_before(d, lIdx) = numel(windows_per_label{lIdx});
        end
    end

    global_min = min(counts_before(:));
    counts_after = counts_before;

    for d = 1:n_folds
        for lIdx = 1:numLabels
            nW = numel(windows_per_dataset{d}{lIdx});
            if nW > global_min
                windows_per_dataset{d}{lIdx} = windows_per_dataset{d}{lIdx}(1:global_min);
                idxRanges_per_dataset{d}{lIdx} = idxRanges_per_dataset{d}{lIdx}(1:global_min);
            end
            counts_after(d, lIdx) = numel(windows_per_dataset{d}{lIdx});
        end
    end
end

function t = build_pairwise_error_table(model_name, test_ds, windowLength, overlap, ...
        fir_order, model_unique_labels, model_label_offset, ...
        n_windows_by_pair, mean_error_by_pair, var_error_by_pair, global_min)

    model_numLabels = numel(model_unique_labels);
    n_pairs = model_numLabels * model_numLabels;

    true_label_raw = zeros(n_pairs, 1);
    true_label = zeros(n_pairs, 1);
    fir_label_raw = zeros(n_pairs, 1);
    fir_label = zeros(n_pairs, 1);
    n_windows = zeros(n_pairs, 1);
    error_mean = nan(n_pairs, 1);
    error_variance = nan(n_pairs, 1);

    row_idx = 0;
    for trueIdx = 1:model_numLabels
        for firIdx = 1:model_numLabels
            row_idx = row_idx + 1;
            true_lbl_val = model_unique_labels(trueIdx);
            fir_lbl_val = model_unique_labels(firIdx);

            true_label_raw(row_idx) = true_lbl_val;
            true_label(row_idx) = true_lbl_val + model_label_offset;
            fir_label_raw(row_idx) = fir_lbl_val;
            fir_label(row_idx) = fir_lbl_val + model_label_offset;
            n_windows(row_idx) = n_windows_by_pair(trueIdx, firIdx);
            error_mean(row_idx) = mean_error_by_pair(trueIdx, firIdx);
            error_variance(row_idx) = var_error_by_pair(trueIdx, firIdx);
        end
    end

    t = table( ...
        repmat({model_name}, n_pairs, 1), ...
        repmat(test_ds, n_pairs, 1), ...
        repmat(windowLength, n_pairs, 1), ...
        repmat(overlap, n_pairs, 1), ...
        repmat(fir_order, n_pairs, 1), ...
        repmat(global_min, n_pairs, 1), ...
        true_label_raw, ...
        true_label, ...
        fir_label_raw, ...
        fir_label, ...
        n_windows, ...
        error_mean, ...
        error_variance, ...
        'VariableNames', {'model_file', 'test_dataset', 'window_length', ...
        'overlap', 'fir_order', 'global_min_windows', ...
        'true_label_raw', 'true_label', 'fir_label_raw', 'fir_label', ...
        'n_windows', 'error_mean', 'error_variance'});
end

function t = build_equalization_table(model_name, test_ds, windowLength, overlap, ...
        fir_order, model_unique_labels, model_label_offset, ...
        counts_before, counts_after, global_min)

    [n_folds, numLabels] = size(counts_before);
    n_rows = n_folds * numLabels;

    dataset_idx = zeros(n_rows, 1);
    label_raw = zeros(n_rows, 1);
    label = zeros(n_rows, 1);
    n_windows_before = zeros(n_rows, 1);
    n_windows_after = zeros(n_rows, 1);

    row_idx = 0;
    for d = 1:n_folds
        for lIdx = 1:numLabels
            row_idx = row_idx + 1;
            label_val = model_unique_labels(lIdx);

            dataset_idx(row_idx) = d;
            label_raw(row_idx) = label_val;
            label(row_idx) = label_val + model_label_offset;
            n_windows_before(row_idx) = counts_before(d, lIdx);
            n_windows_after(row_idx) = counts_after(d, lIdx);
        end
    end

    t = table( ...
        repmat({model_name}, n_rows, 1), ...
        repmat(test_ds, n_rows, 1), ...
        repmat(windowLength, n_rows, 1), ...
        repmat(overlap, n_rows, 1), ...
        repmat(fir_order, n_rows, 1), ...
        repmat(global_min, n_rows, 1), ...
        dataset_idx, ...
        label_raw, ...
        label, ...
        n_windows_before, ...
        n_windows_after, ...
        'VariableNames', {'model_file', 'test_dataset', 'window_length', ...
        'overlap', 'fir_order', 'global_min_windows', ...
        'dataset_idx', 'label_raw', 'label', 'n_windows_before', 'n_windows_after'});
end

function pred_label = predict_window_label(X_win, fir_coeffs_label)
    numChannelsModel = size(fir_coeffs_label, 1);
    if size(X_win, 2) < numChannelsModel
        error('Window channels (%d) < model channels (%d)', size(X_win, 2), numChannelsModel);
    end

    pred_per_channel = zeros(numChannelsModel, 1);

    for ch = 1:numChannelsModel
        B = fir_coeffs_label(ch, :).';
        y_hat = filter(B, 1, X_win(:, ch));
        pred_per_channel(ch) = mean(y_hat);
    end

    pred_label = mean(pred_per_channel);
end

function [windowLength, overlap, fir_order, test_ds, model_label_offset] = ...
        parse_model_metadata(S, model_name, n_folds, default_label_offset)

    token = regexp(model_name, '^FIR_MODEL_(\d+)_([0-9]+(?:\.[0-9]+)?)_(\d+)_([01]+)\.mat$', ...
        'tokens', 'once');

    windowLength = [];
    overlap = [];
    fir_order = [];
    fold_mask = '';

    if isfield(S, 'windowLength'), windowLength = double(S.windowLength); end
    if isfield(S, 'overlap'), overlap = double(S.overlap); end
    if isfield(S, 'fir_order'), fir_order = double(S.fir_order); end
    if isfield(S, 'fold_mask'), fold_mask = S.fold_mask; end
    if isfield(S, 'label_offset')
        model_label_offset = double(S.label_offset);
    else
        model_label_offset = double(default_label_offset);
    end

    if isempty(windowLength) && ~isempty(token), windowLength = str2double(token{1}); end
    if isempty(overlap) && ~isempty(token), overlap = str2double(token{2}); end
    if isempty(fir_order) && ~isempty(token), fir_order = str2double(token{3}); end
    if isempty(fold_mask) && ~isempty(token), fold_mask = token{4}; end

    if isempty(windowLength) || isempty(overlap) || isempty(fir_order)
        error('Cannot parse metadata from model %s', model_name);
    end

    if isfield(S, 'fold')
        test_ds = double(S.fold);
    else
        test_ds = find(fold_mask == '1', 1, 'first');
    end

    if isempty(test_ds)
        error('Cannot detect test fold from model %s', model_name);
    end
    if ~isempty(fold_mask) && numel(fold_mask) ~= n_folds
        warning('fold_mask length mismatch in %s (expected %d)', model_name, n_folds);
    end
end

function fir_matrix = normalize_fir_matrix_shape(fir_matrix_raw, n_labels)
    if ndims(fir_matrix_raw) == 3
        fir_matrix = fir_matrix_raw;
        return;
    end

    size_raw = size(fir_matrix_raw);
    if ndims(fir_matrix_raw) == 2
        if n_labels == 1
            fir_matrix = reshape(fir_matrix_raw, [1, size_raw(1), size_raw(2)]);
            return;
        end
        if size_raw(1) == n_labels
            fir_matrix = reshape(fir_matrix_raw, [n_labels, 1, size_raw(2)]);
            return;
        end
    end

    error('Unsupported FIR matrix shape: %s', mat2str(size_raw));
end

function save_prediction_plot(pred_table, model_name, test_ds, ...
        windowLength, overlap, fir_order, plot_png)

    fig = figure('Visible', 'off');
    if height(pred_table) > 0
        if any(strcmp('fir_label_raw', pred_table.Properties.VariableNames))
            plot_mask = pred_table.true_label_raw == pred_table.fir_label_raw;
        else
            plot_mask = true(height(pred_table), 1);
        end

        if any(plot_mask)
            plot(pred_table.n_window(plot_mask), pred_table.true_label(plot_mask), '.', 'DisplayName', 'True label');
            hold on;
            plot(pred_table.n_window(plot_mask), pred_table.predicted_label(plot_mask), '.', ...
                'DisplayName', 'Predicted label (diag FIR)');
            hold off;
            legend('Location', 'best');
        end
    end
    xlabel('n\_window');
    ylabel('label value');
    title(sprintf('%s | test DS%02d | wl=%d ov=%.2f nb=%d', ...
        model_name, test_ds, windowLength, overlap, fir_order), 'Interpreter', 'none');
    grid on;
    saveas(fig, plot_png);
    close(fig);
end

function t = empty_prediction_table()
    t = table(cell(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
        zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
        zeros(0, 1), zeros(0, 1), ...
        'VariableNames', {'model_file', 'test_dataset', 'window_length', ...
        'overlap', 'fir_order', 'n_window', 'start_idx', ...
        'true_label_raw', 'true_label', 'fir_label_raw', 'fir_label', ...
        'predicted_label', 'prediction_error'});
end

function t = empty_variance_table()
    t = table(cell(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
        zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
        zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
        'VariableNames', {'model_file', 'test_dataset', 'window_length', ...
        'overlap', 'fir_order', 'global_min_windows', ...
        'true_label_raw', 'true_label', 'fir_label_raw', 'fir_label', ...
        'n_windows', 'error_mean', 'error_variance'});
end
