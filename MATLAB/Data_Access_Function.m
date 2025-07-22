function [Data_Select_NOLABEL, Border] = Data_Access_Function(SUBJECT, AMBULATION_Now, Label_Select_Now, CHANNELS, SENSORS, Sample_Num, Sampling_Frequency, Window_Samples, trials)
fprintf('Processing %s %s %s. \n', SUBJECT, AMBULATION_Now, Label_Select_Now);

% f=FileManager('..','PathStructure',{'Subject','Date','Mode','Sensor','Trial'});

% SENSORS = {'gon', 'conditions'};
% allfiles = f.fileList('Subject', SUBJECT, 'Mode', AMBULATION_Now, 'Sensor', SENSORS, 'Trial', '*.mat');
% trials = f.EpicToolbox(allfiles);

% Set up parameters
% Sampling_Frequency = 200; % Hz
% Window_Size = 1; % sec
% Overlap_Size = 0.5; % sec
% 
% Window_Samples = round(Window_Size * Sampling_Frequency);
% Overlap_Samples = round(Overlap_Size * Sampling_Frequency);

% fprintf('%s %s %s files generating. \n', SUBJECT, AMBULATION_Now, Label_Select_Now);

% Initialize target data array:
% HIP_GON_200HZ_Walk = [];
Data_Select_char = [];

% Sampling for all trials in loop:
for i = 1:length(trials)
    trial = trials{i};
    % Initialize Channels as a matrix with the appropriate number of rows and columns
    numRows = size(trial.(SENSORS{1}), 1);
    numCols = numel(CHANNELS);
    Channels = zeros(numRows, numCols);
    % Loop over each channel and populate the corresponding column of Channels
    for j = 1:numCols
        Channels(:,j) = trial.(SENSORS{1}).(CHANNELS{j});
    end
    
    % Do Sampling here
    Channels_Sampled = Channels(1:Sample_Num:end,:);
    
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     % Change the sampling number here if sampling frequency is adjusted!!!
%     Channels_Sampled = Channels(1:5:end,:);
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    LABEL = [trial.conditions.labels.Label];
    LABEL = string(LABEL);
    Channels_Sampled = [Channels_Sampled LABEL];
    
    for rows = 1:length(Channels_Sampled)
        if Channels_Sampled(rows, end) == Label_Select_Now
            Data_Select_char = [Data_Select_char; Channels_Sampled(rows, 1:end-1)];
        end
    end
end

Data_Select_NOLABEL = double(Data_Select_char);

% Find out time stamps at the borders
Time_Stamp = diff(Data_Select_NOLABEL(:, 1));
for i = 1:length(Time_Stamp)                                                       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if abs(Time_Stamp(i)) > 1/Sampling_Frequency + 1/Sampling_Frequency/Sample_Num % Check if this number is proper, here is 0.006
                                                                                   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Time_Stamp(i, 2) = i;
    else
        Time_Stamp(i, 2) = 0;
    end
end

% Find borders for each walk
Border = [1];
for i = 1:length(Time_Stamp)
                                     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if Time_Stamp(i, 2) > Sample_Num % Check if this number is proper, here is 5
                                     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Border=[Border; Time_Stamp(i, 2)+1];
    end
end

% Find time length in rows for each walk
Walk_Length = diff(Border);
Walk_Length = [Walk_Length; length(Data_Select_NOLABEL)-Border(length(Border))];

% Delete data smaller than 1-window time
for i = 1:length(Walk_Length)
    if Walk_Length(i) < Window_Samples
        Data_Select_NOLABEL(Border(i):Border(i)+Walk_Length(i)-1, :) = 0;
    end
end
Data_Select_NOLABEL(all(Data_Select_NOLABEL == 0,2), :) = [];

%%
% % Repeated steps start
Time_Stamp = diff(Data_Select_NOLABEL(:,1));
for i = 1:length(Time_Stamp)
                                                                                   %%%%%%%%%%%%%%%%%%%%
    if abs(Time_Stamp(i)) > 1/Sampling_Frequency + 1/Sampling_Frequency/Sample_Num % Do check as above, should be the same as above
                                                                                   %%%%%%%%%%%%%%%%%%%%
        Time_Stamp(i,2) = i;
    else
        Time_Stamp(i,2) = 0;
    end
end

% Find borders for each walk
Border = [1];
for i = 1:length(Time_Stamp)
                                    %%%%%%%%%%%%%%%%%%%%
    if Time_Stamp(i,2) > Sample_Num % Do check as above, , should be the same as above
                                    %%%%%%%%%%%%%%%%%%%%
        Border=[Border;Time_Stamp(i,2)+1];
    end
end

% Find time length in rows for each walk
Walk_Length = diff(Border);
Walk_Length = [Walk_Length; length(Data_Select_NOLABEL)-Border(length(Border))];
% % Repeated steps end
end
