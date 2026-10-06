function File = GenerateSample(Folder, Seed)

if nargin < 1 || isempty(Folder)
    Folder = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'dataset');
end
if nargin < 2
    Seed = 20261006;
end
Data = DataSimulate('Seed', Seed);
Data.SubjectID = arrayfun(@(Index) sprintf('SYN_EMGPN_%04d', Index), (1:numel(Data.Ydata))', 'UniformOutput', false);
if exist(Folder, 'dir') ~= 7
    mkdir(Folder);
end
File = fullfile(Folder, 'sample.csv');
DataWrite(Data, File);

end
