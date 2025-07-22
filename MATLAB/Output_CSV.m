function Output_CSV(SUBJECT, AMBULATION_Now, Label_Select_Now, CHANNELS, SENSORS, Segment_All_Trials_y_labels, Segment_All_Trials_Channels, filepath_X, filepath_y)

% Set up output path, if folders don't exist then make new folders
if ~exist(filepath_X, 'dir')
    mkdir(filepath_X)
end
if ~exist(filepath_y, 'dir')
    mkdir(filepath_y)
end

numChannels = length(CHANNELS);
Sensor_Name = SENSORS{1};

for iChannel = 2:numChannels
    
    var_name = ['Segment_All_Trials_Channels.Segment_All_Trials_' CHANNELS{iChannel}];
    
    filename_X = sprintf('%s_%s_%s_%s_%s_X_Windows.csv', SUBJECT, AMBULATION_Now, SENSORS{1}, Label_Select_Now, CHANNELS{iChannel});
    
%     filename_1 = sprintf('%s_%s_%s_%s_Windows.csv', SUBJECT, AMBULATION, Label_Select, channel{1});
%     filename_2 = sprintf('%s_%s_%s_%s_Windows.csv', SUBJECT, AMBULATION, Label_Select, channel{2});
    
    fullpath_X = fullfile(filepath_X, filename_X);
%     fullpath_2 = fullfile(filepath_X, filename_2);
    
    writematrix(eval(var_name), fullpath_X);
%     writematrix(Segment_All_Trials_Hip_Sagittal, fullpath_2);
    
    fprintf('%s %s %s %s %s X file is completed. \n', SUBJECT, AMBULATION_Now, SENSORS{1}, Label_Select_Now, CHANNELS{iChannel});
    
end

filename_y = sprintf('%s_%s_%s_%s_y_labels.csv', SUBJECT, AMBULATION_Now, SENSORS{1}, Label_Select_Now);
fullpath_y = fullfile(filepath_y, filename_y);

writematrix(Segment_All_Trials_y_labels, fullpath_y);

fprintf('%s %s %s %s y file is completed. \n', SUBJECT, AMBULATION_Now, SENSORS{1}, Label_Select_Now);
end

