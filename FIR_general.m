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

X_path = fullfile(script_dir, 'data', macro_folder, test, X_file);

X_file = [finger '_downsampled.npy']; 
X_path = fullfile(script_dir, 'data', macro_folder, test, test_type, [test_type '_' test_n], X_file);
Y_path = fullfile(script_dir, 'data', macro_folder, test, test_type, [test_type '_' test_n], 'sensor_values_downsampled.npy');


% Load data
X = Neuropixel.readNPY(X_path);   % shape: [n_samples, 8]
Y = Neuropixel.readNPY(Y_path);   % shape: [n_samples, 1]

Y = abs(Y(:,3));

% Check matching sizes
assert(size(X,1) == length(Y), 'X and y must have the same number of samples');

% FIR model parameters
fir_order = 50;  % Number of taps (nb)
nk = 1;          % Input delay (1 sample)
na = 0;          % No autoregressive component

numChannels = size(X,2);
firModels = cell(1, numChannels);  % Store one model per input channel

% Loop over each channel and fit FIR model
for ch = 1:numChannels
    u = X(:, ch);  % Current input channel

    % Create iddata object: y is output, u is input
    data_id = iddata(Y, u, 1);  % Ts = 1 (arbitrary if unit sample spacing)

    % Define ARX model structure [na nb nk]
    modelOrders = [na fir_order nk];

    % Fit FIR model using ARX
    model = arx(data_id, modelOrders);
    corr_val = corr(u, Y);
    fprintf('Corr(Channel %d): %.4f\n',ch, corr_val);

    % Store model in cell array
    firModels{ch} = model;
    y_pred = predict(model, data_id);
    y_pred = y_pred.OutputData;  % Extract numeric array from iddata
    % Plot
    figure;
    plot(Y, 'LineWidth', 1.5); hold on;
    plot(y_pred, '--', 'LineWidth', 1.5);
    legend('True y', 'Predicted y');
    xlabel('Time (samples)');
    ylabel('Output');
    title(sprintf('Label %d – Channel %d', label, ch));
    grid on;
    drawnow;
end

% Display confirmation
fprintf('FIR models successfully trained for %d channels.\n', numChannels);

% Save the models
savePath_Model = fullfile(script_dir, 'data', macro_folder, test, 'firModels.mat');
% save(savePath_Model, 'firModels');
% 
% fprintf('FIR models saved to: %s\n', savePath_Model);
