function Feature_Extraction(SUBJECT, AMBULATION_Now, Label_Select_Now, CHANNELS, SENSORS, Segment_All_Trials_Channels, filepath_Feature_Extraction)
% Set up output path, if folders don't exist then make new folders
if ~exist(filepath_Feature_Extraction, 'dir')
    mkdir(filepath_Feature_Extraction)
end

numChannels = length(CHANNELS);
Sensor_Name = SENSORS{1};

for iChannel = 2:numChannels
    % Name of the matrix (windows) waiting for generating features
    var_name = ['Segment_All_Trials_Channels.Segment_All_Trials_' CHANNELS{iChannel}];
    
    Windows = eval(var_name);
    [Num_Rows, Num_Columns] = size(Windows);
    Features_Matrix = zeros(Num_Rows, 5);
    
    for row = 1:Num_Rows
        Current_Row = Windows(row, :);
        
        Mean_Row = mean(Current_Row);
        
        Abs_Diff = abs(Current_Row - Mean_Row);
        
        Mean_Abs_Dev = mean(Abs_Diff);
        
        Std_Dev = std(Current_Row);
        
        Min_Value = min(Current_Row);
        
        Max_Value = max(Current_Row);
        
        Last_Value = Current_Row(end);
        
        Features_Matrix(row, :) = [Mean_Abs_Dev, Std_Dev, Min_Value, Max_Value, Last_Value];
    end
    
    filename_X = sprintf('%s_%s_%s_%s_%s_X_Features.csv', SUBJECT, AMBULATION_Now, Sensor_Name, Label_Select_Now, CHANNELS{iChannel});
    fullpath_X = fullfile(filepath_Feature_Extraction, filename_X);
    writematrix(Features_Matrix, fullpath_X);
    % disp(Features_Matrix);
end
end

