% clc;
% clear;

%% 
init();def=defaults;

%%
% Select subject
%%%%%% Caution %%%%%%
% Comment this part when using BATCHRUN
% SUBJECT = 'AB17';

%%
% Select ambulation modes
% AMBULATION={'levelground','ramp','stair','treadmill'};
AMBULATION = {'levelground','ramp','stair'};

%%
% Select sensors
% SENSORS={'markers','gcLeft','gcRight','conditions','ik','id','emg','imu','gon','jp'};
% 'conditions' is necessary, that's where labels are stored
% So far only 1 sensor is supported since the sampling frequency of each sensor is different 
SENSORS = {'gon','conditions'};

%%
% Select channels
% Select specific channels according to your selected sensors and the Headers in the dataset
% Caution on the format! Please check the exact channel names in the dataset
% The channel "Header" is necessary.
% e.g. CHANNELS = {'Header', 'hip_frontal ', 'hip_sagittal'};
CHANNELS = {'Header', 'hip_frontal', 'hip_sagittal'};

%% Load this subjects ambulation data
fprintf('Processing files for subject %s\n',SUBJECT);

f=FileManager('..','PathStructure',{'Subject','Date','Mode','Sensor','Trial'});
allfiles = f.fileList('Subject', SUBJECT, 'Mode', AMBULATION, 'Sensor', SENSORS, 'Trial', '*.mat');
trials=f.EpicToolbox(allfiles);

%%
% All labels that are provided:
% You don't need to change this
Labels_levelground = {'stand', 'stand-walk', 'walk', 'turn1', 'turn2', 'walk-stand'};
Labels_ramp = {'walk-rampascent', 'rampascent', 'rampascent-walk', 'walk-rampdescent', 'rampdescent', 'rampdescent-walk'};
Labels_stair = {'walk-stairascent', 'stairascent', 'stairascent-walk', 'walk-stairdescent', 'stairdescent', 'stairdescent-walk'};

% Select your labels:
% Labels_Select = {'stand','walk','rampascent','rampdescent','stairascent','stairdescent'};
Labels_Select = {'stand', 'stand-walk', 'walk', 'turn1', 'turn2', 'walk-stand', ...
    'walk-rampascent', 'rampascent', 'rampascent-walk', 'walk-rampdescent', 'rampdescent', 'rampdescent-walk', ...
    'walk-stairascent', 'stairascent', 'stairascent-walk', 'walk-stairdescent', 'stairdescent', 'stairdescent-walk'};
% Remember the indexs of the labels when they are related to indexs in numbers
% Here stand = 1, walk = 2, etc.

%%
% Find intersections between selected labels and label sets
% for i = 1:length(AMBULATION)
%     current_ambulation = AMBULATION{i};
%     current_labels = eval(sprintf('Labels_%s', current_ambulation));
%     intersection = intersect(Labels_Select, current_labels);
%     fprintf('For %s, the intersection is: \n', current_ambulation);
%     disp(intersection);
% end

%%
% Set up sampling parameters
Sampling_Frequency = 200; % Hz
Window_Size = 1; % sec
Overlap_Size = 0.5; % sec

%%
% Set up sample number: Sample_Num
% e.g. Sampling frequency of goniometer is 1kHz, if resampled to 200Hz, then Sample_Num = 5
Sample_Num = 5;

%%
% Set up windows
Window_Samples = round(Window_Size * Sampling_Frequency);
Overlap_Samples = round(Overlap_Size * Sampling_Frequency);

%%
% Set up output path
filepath_X = ['D:\Research\CAMARGO_ET_AL_J_BIOMECH_DATASET\scripts\' SUBJECT '_Windows\'];
filepath_y = ['D:\Research\CAMARGO_ET_AL_J_BIOMECH_DATASET\scripts\' SUBJECT '_y_labels\'];

%%
for Ambulation_Index = 1:numel(AMBULATION)
    
    % Find intersections between selected labels and label sets
    current_ambulation = AMBULATION{Ambulation_Index};
    current_labels = eval(sprintf('Labels_%s', current_ambulation));
    Labels_Intersection = intersect(Labels_Select, current_labels);
    fprintf('For %s, the intersection is: \n', current_ambulation);
    disp(Labels_Intersection);
    % If the intersection is empty:
    if isempty(Labels_Intersection)
        disp('Label cell is empty');
        
    else
        for Label_Index = 1:numel(Labels_Intersection)
            AMBULATION_Now = cell2char(AMBULATION(Ambulation_Index));
            Label_Select_Now = cell2char(Labels_Intersection(Label_Index));
            fprintf('Now generating [%s] in ambulation [%s].\n',Label_Select_Now, AMBULATION_Now);
            %%
            [Data_Select_NOLABEL, Border] = Data_Access_Function(SUBJECT, AMBULATION_Now, Label_Select_Now, CHANNELS, SENSORS, Sample_Num, Sampling_Frequency, Window_Samples, trials);
            fprintf('Data Generation for [%s] is Completed.\n', Label_Select_Now);
            fprintf('Now Processing Window Segmentation for [%s].\n', Label_Select_Now);
            %%
            [Segment_All_Trials_Channels] = Generate_X_Windows(Data_Select_NOLABEL, Border, Window_Samples, Overlap_Samples, CHANNELS);
            fprintf('Window Segmentation for [%s] is completed.\n', Label_Select_Now);
            fprintf('Now generating Labels for [%s].\n', Label_Select_Now);
            [Segment_All_Trials_y_labels] = Generate_y_Labels(Segment_All_Trials_Channels, Labels_Select, Labels_Intersection, Label_Index);
            fprintf('Label generation for [%s] is completed.\n', Label_Select_Now);
            disp('');
            
            %%
            Output_CSV(SUBJECT, AMBULATION_Now, Label_Select_Now, CHANNELS, SENSORS, Segment_All_Trials_y_labels, Segment_All_Trials_Channels, filepath_X, filepath_y);
        end
        
    end
    
end

fprintf('Subject %s data generation is completed.', SUBJECT);