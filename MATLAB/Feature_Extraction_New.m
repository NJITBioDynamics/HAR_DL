function Feature_Extraction_New(SUBJECT, CHANNELS, SENSORS, X_Windows, filepath_Feature_Extraction)
% Set up output path, if folders don't exist then make new folders
if ~exist(filepath_Feature_Extraction, 'dir')
    mkdir(filepath_Feature_Extraction)
end

numChannels = length(CHANNELS);
Sensor_Name = SENSORS{1};

for iChannel = 2:numChannels
    % Name of the matrix (windows) waiting for generating features
    var_name = ['X_Windows.' CHANNELS{iChannel} '_X_Windows'];
    
    Windows = eval(var_name);
    [Num_Rows, Num_Columns] = size(Windows);
    Features_Matrix = zeros(Num_Rows, 5);
    
    for row = 1:Num_Rows
        Current_Row = Windows(row, :);
        
        Mean_Row = mean(Current_Row);
        
        Abs_Diff = abs(Current_Row - Mean_Row);
        
        Mean_Abs_Dev = mean(Abs_Diff); % MAD
        
        Std_Dev = std(Current_Row); % STD
        
        Min_Value = min(Current_Row); % Min
        
        Max_Value = max(Current_Row); % Max
        
        Last_Value = Current_Row(end); % Last Value
        
        Features_Matrix(row, :) = [Mean_Abs_Dev, Std_Dev, Min_Value, Max_Value, Last_Value];
    end
    
    filename_X = sprintf('%s_%s_%s_X_Features.csv', SUBJECT, Sensor_Name, CHANNELS{iChannel});
    fullpath_X = fullfile(filepath_Feature_Extraction, filename_X);
    writematrix(Features_Matrix, fullpath_X);
    % disp(Features_Matrix);
end
fprintf('Feature Extraction is Completed. \n');
end