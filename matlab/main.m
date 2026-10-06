function [Result, ModelInit] = main(varargin)

OriginalPath = path;
Cleanup = onCleanup(@() path(OriginalPath));
addpath(fullfile(fileparts(mfilename('fullpath')), 'model'));
[Result, ModelInit] = RunEMGPN(varargin{:});

end
