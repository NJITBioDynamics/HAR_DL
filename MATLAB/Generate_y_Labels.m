function [Segment_All_Trials_y_labels] = Generate_y_Labels(Segment_All_Trials_Channels, Labels_Select, Labels_Intersection, Label_Index)

% Check if all fields have the same size
field_names = fieldnames(Segment_All_Trials_Channels);
sizes = cellfun(@(fn) size(Segment_All_Trials_Channels.(fn)), field_names, 'UniformOutput', false);
if ~isequal(sizes{:})
    error('Field sizes are not equal!');
end

FirstField = Segment_All_Trials_Channels.(field_names{1});
idx = find(strcmp(Labels_Select, Labels_Intersection(Label_Index)));
Num_Rows = size(FirstField, 1);
Segment_All_Trials_y_labels = ones(Num_Rows, 1) * idx;
    
end