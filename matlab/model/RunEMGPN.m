function [Result, ModelInit] = RunEMGPN(DataFile, Parameter, OutputFolder)

if nargin < 1 || isempty(DataFile)
    DataFile = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'dataset', 'sample.csv');
end
if nargin < 2 || isempty(Parameter)
    Parameter = struct();
end
if nargin < 3
    OutputFolder = '';
end
Default = struct('NumIter', 1, 'NumFold', 5, 'MaxEpoch', 20, 'NumPermute', 2, 'Seed', 0);
Names = fieldnames(Default);
for Index = 1:numel(Names)
    if ~isfield(Parameter, Names{Index})
        Parameter.(Names{Index}) = Default.(Names{Index});
    end
end
Dataset = DataRead(DataFile, 'ClassName', {'SCD','MCI','AD','VCI','VD'});
[Result, ModelInit] = CrossValidation(Dataset, Parameter);
ResultReport(Result);
if ~isempty(OutputFolder)
    ResultExport(Result, OutputFolder);
end

end
