%%%%%% Caution %%%%%%
% Comment this part when using BATCHRUN
clc;
clear;

%% 
init();def=defaults;

%%
% Select subject
%%%%%% Caution %%%%%%
% Comment this part when using BATCHRUN
SUBJECT = 'AB17';

%%
% Select ambulation modes
% AMBULATION={'levelground','ramp','stair','treadmill'};
AMBULATION = {'levelground','ramp','stair'};

%%
SENSORS = {'ik','conditions'};

%%
% CHANNELS = {'Header', 'ankle_sagittal', 'ankle_frontal', 'knee_sagittal', 'hip_sagittal', 'hip_frontal'};
CHANNELS = {'Header', 'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l'};

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
Labels_Select = {'stand','walk','rampascent','rampdescent','stairascent','stairdescent'};
% Labels_Select = {'stand', 'stand-walk', 'walk', 'turn1', 'turn2', 'walk-stand', ...
%     'walk-rampascent', 'rampascent', 'rampascent-walk', 'walk-rampdescent', 'rampdescent', 'rampdescent-walk', ...
%     'walk-stairascent', 'stairascent', 'stairascent-walk', 'walk-stairdescent', 'stairdescent', 'stairdescent-walk'};
% Remember the indexs of the labels when they are related to indexs in numbers
% Here stand = 1, walk = 2, etc.

%%
for i = 1:numel(trials)
    trial = trials{i};
    fprintf('Current trial is %s  \n', trials{i ,1}.info.Trial);
    Trial_IK = trial.ik;
    Trial_IK_Channels_Selected = Trial_IK(:,{'Header','hip_flexion_r', 'hip_adduction_r','hip_rotation_r','knee_angle_r','ankle_angle_r', 'hip_flexion_l','hip_adduction_l','hip_rotation_l','knee_angle_l','ankle_angle_l'});
    Trial_IK_Double = table2array(Trial_IK_Channels_Selected);
    [Num_Rows, Num_Columns] = size(Trial_IK_Double);
    fprintf('Size of the Trial is %d * %d \n', Num_Rows, Num_Columns);
    
    % Extract .mat filename
%     Filepath_trials = trials{i, 1}.conditions.file;
%     [~, Filename, Filext] = fileparts(Filepath_trials);
    File_Name = trials{i, 1}.info.Trial;
    disp(File_Name);
    
    Header = Trial_IK_Double(:, 1);
    
    Angular_Velocity = [];
    for j = 2:Num_Columns
        Angular_Vel = Calculate_Angular_Velocity(Trial_IK_Double(:, j), Header);
        Angular_Velocity = [Angular_Velocity, Angular_Vel];
    end
    [Num_Rows_AV, Num_Columns_AV] = size(Angular_Velocity);
    fprintf('Size of the Angular_Velocity is %d * %d \n', Num_Rows_AV, Num_Columns_AV);
    
    % Column_Names = {'ankle_sagittal_angvel', 'ankle_frontal_angvel', 'knee_sagittal_angvel', 'hip_sagittal_angvel', 'hip_frontal_angvel'};
    Column_Names = {'hip_flexion_r_angvel', 'hip_adduction_r_angvel','hip_rotation_r_angvel', 'knee_angle_r_angvel', 'ankle_angle_r_angvel', 'hip_flexion_l_angvel', 'hip_adduction_l_angvel','hip_rotation_l_angvel', 'knee_angle_l_angvel', 'ankle_angle_l_angvel'};
    Angular_Velocity_Table = array2table(Angular_Velocity, 'VariableNames', Column_Names);
    
    data = [Trial_IK, Angular_Velocity_Table];
    
    % Update trial here
    % trials{i}.ik = Combined_IK;
    
    Filepath_allfiles = allfiles{i};
    [~, Filename_allfiles, Filext_allfiles] = fileparts(Filepath_allfiles);
    File_Name_allfiles = [Filename_allfiles Filext_allfiles];
    disp(File_Name_allfiles);

    % Access .mat filename
%     Filepath_trials = trials{i, 1}.conditions.file;
%     [~, Filename, Filext] = fileparts(Filepath_trials);
    File_Name = trials{i, 1}.info.Trial;
    disp(File_Name);    

    % Access the path of the current .mat file
    File_Index = cellfun(@(x) contains(x, File_Name), allfiles);
    Matched_Path = allfiles{File_Index};
    fprintf('The matched file path is: %s', Matched_Path);

    disp(' ');

    save(Matched_Path, "data")

%     test_filepath = 'D:\Research\Dataset_Test\levelground_ccw_fast_01_01.mat';
    
%     if strcmp(File_Name, File_Name_allfiles)        
%         save(Filepath_allfiles, 'Combined_IK', "-append")
% %         save(test_filepath, 'data');
%         fprintf('Subject %s Angular Velocity Generated %d/%d', SUBJECT, i, numel(trials));
%         disp(' ');
%     else
%         error('The strings File_Name and File_Name_allfiles are not the same.')
%     end
end





 
