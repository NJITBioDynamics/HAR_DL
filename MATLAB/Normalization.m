function Normalized_X_Windows = Normalization(CHANNELS, Channels_X, X_Windows)
% normalize_signals Normalizes an m by n matrix of signals to a range of [-1, 1]
numChannels = length(CHANNELS);
% Sensor_Name = SENSORS{1};

Normalized_X_Windows = struct();
for i = 2:numChannels
    Normalized_X_Windows.(Channels_X{i-1}) = [];
end

for iChannel = 2:numChannels
    var_name = ['X_Windows.' CHANNELS{iChannel} '_X_Windows'];
    Windows = eval(var_name);
    
    Normalized_Windows = (Windows - min(Windows(:))) / (max(Windows(:)) - min(Windows(:)));
    
    Normalized_Windows = 2 * Normalized_Windows -1;
    Normalized_X_Windows.(Channels_X{iChannel-1}) = Normalized_Windows;
end
end