function data = csv2struct(filename)
% Loads a CSV file to a structure
% 
% Author: A. Conrad Nied (anied@cs.washington.edu)
%
% Changelog:
% 2013-11-01 Created, based on GPS1.8/struct2csv and QualsPlan/peopleProcess

% Open the file
fid = fopen(filename);

% Get first line (fields)
tline = fgetl(fid);
breaks = [0 find(tline==',') length(tline)+1];
N_fields = length(breaks) - 1;
fields = cell(N_fields, 1);
for i = 1:N_fields
    field = tline((breaks(i) + 1):(breaks(i+1) - 1));
    field(field==' ') = '_';
    fields{i} = field;
end

% Get entries
tline = fgetl(fid);
N_entries = 0;
while ischar(tline)
    breaks = [0 find(tline==',') length(tline)+1];
    entryline = [];
    for i = 1:N_fields
        field = fields{i};
        entry = tline((breaks(i) + 1):(breaks(i+1) - 1));
%         fprintf('%d\t%s\t%s', N_entries, field, entry);
        if(mean(isstrprop(entry, 'digit') | isstrprop(entry, 'punct'))==1)
            entryline.(field) = str2double(entry);
        else
            entryline.(field) = entry;
        end
    end
    
    N_entries = N_entries + 1;
    data(N_entries) = entryline; %#ok<AGROW>
    
    tline = fgetl(fid);
end

fclose(fid);

end % fucntion csv2struct

% % Load the file
% data = importdata(filename);
% 
% % Get the field names
% line = data{1};
% commas = [0 find(line == ',') length(line) + 1];
% N_fields = length(commas - 1);
% fields = cell(N_fields, 1);
% for i_field = 1:N_fields
%     fields{i} = line((commas(i)) + 1):(commas(i+1) - 1));
% end
% 
% 
% % Open the file
% fileID = fopen(filename, 'w');
% 
% % Write the field names
% struct_fields = fields(struct);
% 
% for i_field = 1:length(struct_fields);
%     if(i_field > 1)
%         fprintf(fileID, ', ');
%     end
%     fprintf(fileID, struct_fields{i_field});
% end
% 
% % Write entries
% for i_entry = 1:length(struct)
%     fprintf(fileID, '\n');
%     for i_field = 1:length(struct_fields);
%         if(i_field > 1)
%             fprintf(fileID, ', ');
%         end
%         
%         entry = struct(i_entry).(struct_fields{i_field});
%         if(isnumeric(entry) || islogical(entry))
%             entry = num2str(entry);
%         elseif(iscell(entry))
%             entry = gpse_convert_string(entry);
%             entry(entry == ',') = ';';
%         end
%         fprintf(fileID, entry);
%     end
% end
% 
% % Close the file
% fclose(fileID);