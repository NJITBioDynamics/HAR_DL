# Codes for the paper: Optimizing Sensor and Data Selection on Lower Limbs via Deep Learning for Real-Time Human Activity Recognition
The goal of this repository is ro generate segmented data for selected labeled activities and selected sensor channels for all trials, selected ambulations, and selected subjects.

This repository should be downloaded and run together with the EpicToolbox by EPIC Lab, GaTech.

The script and functions are suitable for Lower Limb Dataset, EPIC Lab, GaTech.
Dataset link: https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/

Steps before running:
- Run /scripts/EpicToolbox/install.m and /scripts/MoCapTools/install.m
- Install the following add-ons in MATLAB:
	1.Signal Processing Toolbox
	2.Function: cell2char
	
*************************************************************************************************************************************************
**Bug notice on the Lower Limb Dataset: please rename all .mat file to lowercase letters, or it will cause a bug when overwriting the matrices.**
*************************************************************************************************************************************************
The bug above has been fixed on 09/01/2023. The solution is not perfect, but it does not cause any error, and the cost is small.

For the dataset itself, now angular velocity of frontal and sagittal angles has been added under 'gon' folder.
Now angular velocity of IK data is updated under 'ik' folder. (09/05/2023)

2 main script and 4 functions to generate segmented activity data and labels in CSV files.
- "BATCHRUN.m" is to run "Data_Access_Main.m" in batches of subjects. Run this script to generate Xs and ys.
- "Data_Access_Main.m" is the main script for setups. 
- "Data_Access_Function.m" is to generate time series for all channels selected and it also returns the border positions (in rows) of each labels.
- "Generate_X_Windows.m" returns a MATLAB struct contains segmented windows data for all channels. Now data of all ambulations are stacked in 1 file.
- "Generate_y_Labels.m" returns the corresponding label indices for segmented windows data above. Now lables of all ambulations are stacked in 1 file.
- "Output_CSV_X_Windows.m" and "Output_CSV_y_Labels.m" are to generate .csv format data files for Xs and ys from the above. The naming format of the output files is: 
    X: "SUBJECT_SENSOR_LABEL_CHANNEL_X_Windows.csv" 
    y: "SUBJECT_SENSOR_LABEL_y_labels.csv"
- Added a feature extraction function "Feature_Extraction_New.m". So far the features provided are: MAD, STD, Min, Max and the last value of a window.
- Added a normalization function "Normalization.m". This function is only used in batch running to generate data of multiple subjects.

Extra script:
- "Combine_Files.m" is to stack windows and labels among subjects. It will also separate the dataset to apply cross-validation. 

Extra functions:
- 'Separate_Gravity.m' and 'Separate_Gravity_ENMO.m' provide 3 metrics: ENMO, HFEN and HFEN+ to remove gravity part in accelaration in 'imu' set. These 2 functions need to be cooperated with the 2 filter functions "butterworth_high_pass.m" and 'butterworth_low_pass.m'.

How to use the dataset and this respository:
The dataset contains 22 zipped package subject data, and you may need to download the data 1 by 1. Apart from the subject data there are 2 other files and 1 package, "README.txt", "SubjectInfo.mat", and "script".

The script package is the EpicToolbox provided by EPIC Lab. More information about EpicToolbox: https://github.com/JonathanCamargo/EpicToolbox

After downloading all the files mentioned above, unzip all packages in the same folder. Then this folder will contain folders for all subject data, script folder, "README.txt" and "SubjectInfo.mat".

Download this respsitory in the "script" folder.

There is another "BATCHRUN.m" provided in the folder "script". Use this one in the resporitory to replace the provided one.

Settings:
- The setting for subjects is in BATCHRUN.m. You can select any subject to generate data.
- The rest of the settings are in Data_Access_Main.m

The path for output files are customed in Data_Access_4.m.

