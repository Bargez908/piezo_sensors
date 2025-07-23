% Base directory
script_dir = pwd;

% Define subfolders and parameters
macro_folder = 'records_final';
test = 'thumb_pressure';
finger = 'thumb';
test_type = 'pressure';   % or 'sliding'
test_n = '0';

% Construct file paths
X_file = [finger '_concatenated.npy'];       % Input: piezo
label_file = 'labels.npy';                        % Output: labels

X_path = fullfile(script_dir, 'data', macro_folder, test, X_file);
labels_path = fullfile(script_dir, 'data', macro_folder, test, test_type, [test_type '_' test_n], label_file);

X_file = [finger '_downsampled.npy']; 
X_path = fullfile(script_dir, 'data', macro_folder, test, test_type, [test_type '_' test_n], X_file);
Y_path = fullfile(script_dir, 'data', macro_folder, test, test_type, [test_type '_' test_n], 'sensor_values_downsampled.npy');


% Load data
X = Neuropixel.readNPY(X_path);   % shape: [n_samples, 8]
labels = double(Neuropixel.readNPY(labels_path)); % ensure double
Y = Neuropixel.readNPY(Y_path);
% Obain Z component from Y
Y = abs(Y(:,3));

% Check matching sizes
assert(size(X,1) == length(Y), 'X and y must have the same number of samples');

% FIR model parameters
fir_order = 10;  % Number of taps
nk = 0;          % Input delay
na = 0;          % No AR component

numChannels = size(X,2);

% Get unique labels
unique_labels = unique(labels);
numLabels = length(unique_labels);

% Cell array: rows = labels, cols = channels
firModels = cell(numLabels, numChannels);

% Loop over labels
for lIdx = 1:numLabels
    label = unique_labels(lIdx);

    % Extract data for this label
    idx = (labels == label);
    X_label = X(idx, :);
    Y_label = Y(idx);

    fprintf('Training FIR models for label %d (samples: %d)\n', label, sum(idx));

    % Loop over each channel for this label
    for ch = 1:numChannels
        u = X_label(:, ch);

        % Create iddata object for this label
        data_id = iddata(Y_label, u, 1);

        % Define ARX orders
        modelOrders = [na fir_order nk];

        % Fit FIR model
        model = arx(data_id, modelOrders);

        % Store model
        firModels{lIdx, ch} = model;
        corr_val = corr(u, Y_label);
        fprintf('Corr(Channel %d, label %d): %.4f\n',ch,label, corr_val);
        % check output
        y_pred = predict(model, data_id);
        y_pred = y_pred.OutputData;  % Extract numeric array from iddata
        % Plot
        figure;
        plot(Y_label, 'LineWidth', 1.5); hold on;
        plot(y_pred, '--', 'LineWidth', 1.5);
        legend('True y', 'Predicted y');
        xlabel('Time (samples)');
        ylabel('Output');
        title(sprintf('Label %d – Channel %d', label, ch));
        grid on;
        drawnow;
    end
end

% Display confirmation
fprintf('FIR models successfully trained for %d labels and %d channels.\n', numLabels, numChannels);

% Save models
savePath_Model = fullfile(script_dir, 'data', macro_folder, test, 'firModels_byLabel.mat');
% save(savePath_Model, 'firModels', 'unique_labels');
