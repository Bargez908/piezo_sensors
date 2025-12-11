% Base directory (you can use pwd if you want to start from current folder)
script_dir = pwd;  % Equivalent to os.path.dirname(__file__) if you're running from the script's folder

% Define subfolders and parameters
macro_folder = 'records_final';
test = 'thumb_pressure';
test_n = 4;
test_type = 'pressure';   % or 'sliding'
finger = 'thumb';  % 'thumb', 'index', 'middle', 'ring', 'little'

max_overlap = true; % Set to true for maximum overlap

% Construct file paths dynamically
fileName = [finger '_concatenated.npy'];
piezo_data_path = fullfile(script_dir, 'data', macro_folder, test, fileName);

data = Neuropixel.readNPY(piezo_data_path); 

% Parameters
fs = 313; % Sampling frequency
%windowLength = fs; % 1 second window
windowLength = 80; 
overlap = 0.99; % 80% overlap
stepSize = floor(windowLength * (1 - overlap)); % Step size for windowing
if max_overlap
    stepSize = 1; % Maximum overlap
end
order = 5; % AR model order

% Number of channels and total timepoints
[numSamples, numChannels] = size(data);

% Compute number of windows
numWindows = floor((numSamples - windowLength) / stepSize) + 1;

% Initialize 3D array for AR coefficients and 2D for noise variances
arCoeffs = zeros(numChannels, order + 1, numWindows);
noiseVariances = zeros(numChannels, numWindows);

% Loop through each channel
for ch = 1:numChannels
    channelData = data(:, ch);

    windowIdx = 1;
    for startIdx = 1:stepSize:(length(channelData) - windowLength + 1)
        windowData = channelData(startIdx:startIdx + windowLength - 1);

        % Estimate AR parameters using Burg's method
        [coeffs, noiseVar] = arburg(windowData, order);

        % Store the results
        arCoeffs(ch, :, windowIdx) = coeffs;
        noiseVariances(ch, windowIdx) = noiseVar;

        windowIdx = windowIdx + 1;
    end
end

% Display dimensions
fprintf('AR Coefficients size: %s\n', mat2str(size(arCoeffs))); % [channels x order+1 x windows]
fprintf('Noise Variances size: %s\n', mat2str(size(noiseVariances))); % [channels x windows]

savePath_AR = fullfile(script_dir, 'data', macro_folder, test, 'arCoeffs.npy');
savePath_Noise = fullfile(script_dir, 'data', macro_folder, test, 'noiseVariances.npy');


figure; % Create a new figure
plot(noiseVariances(1, :)); % Plot the first row
xlabel('Index'); % Label for x-axis
ylabel('Variance'); % Label for y-axis
title('Plot of the First Row of noiseVariance'); % Title of the plot
grid on;

% Save the arrays
Neuropixel.writeNPY(arCoeffs, savePath_AR);
Neuropixel.writeNPY(noiseVariances, savePath_Noise);