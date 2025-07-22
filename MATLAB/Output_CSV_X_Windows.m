function Output_CSV_X_Windows(SUBJECT, CHANNELS, SENSORS, filepath_X, X_Windows)
% Set up output path, if folders don't exist then make new folders
if ~exist(filepath_X, 'dir')
    mkdir(filepath_X)
end

numChannels = length(CHANNELS);
Sensor_Name = SENSORS{1};

for iChannel = 2:numChannels
%     var_name = ['Segment_All_Trials_Channels.Segment_All_Trials_' CHANNELS{iChannel}];
    var_name = ['X_Windows.' CHANNELS{iChannel} '_X_Windows'];
    
    filename_X = sprintf('%s_%s_%s_X_Windows.csv', SUBJECT, Sensor_Name, CHANNELS{iChannel});
    
%     filename_1 = sprintf('%s_%s_%s_%s_Windows.csv', SUBJECT, AMBULATION, Label_Select, channel{1});
%     filename_2 = sprintf('%s_%s_%s_%s_Windows.csv', SUBJECT, AMBULATION, Label_Select, channel{2});
    
    fullpath_X = fullfile(filepath_X, filename_X);
%     fullpath_2 = fullfile(filepath_X, filename_2);
    
    writematrix(eval(var_name), fullpath_X);
%     writematrix(Segment_All_Trials_Hip_Sagittal, fullpath_2);
    
    fprintf('%s %s %s X file is completed. \n', SUBJECT, Sensor_Name, CHANNELS{iChannel});
    
end

end