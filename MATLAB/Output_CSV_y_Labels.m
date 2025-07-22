function Output_CSV_y_Labels(SUBJECT, CHANNELS, SENSORS, filepath_y, y_Labels)
% Set up output path, if folders don't exist then make new folders
if ~exist(filepath_y, 'dir')
    mkdir(filepath_y)
end

numChannels = length(CHANNELS);
Sensor_Name = SENSORS{1};

for iChannel = 2:numChannels
    
    var_name = ['y_Labels.' CHANNELS{iChannel} '_y_Labels'];
    
    filename_y = sprintf('%s_%s_%s_y_labels.csv', SUBJECT, Sensor_Name, CHANNELS{iChannel});
    fullpath_y = fullfile(filepath_y, filename_y);
    
    writematrix(eval(var_name), fullpath_y);
    
end
end