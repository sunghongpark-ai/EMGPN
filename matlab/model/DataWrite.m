function DataWrite(Dataset, File, varargin)

Opt = struct('Precision', 4, 'IdColumn', 'SubjectID', 'LabelColumn', 'Diagnosis');
if mod(numel(varargin), 2) ~= 0
    error('EMGPN:InvalidInput', 'Options must be name-value pairs.');
end
for i = 1:2:numel(varargin)
    if ~isfield(Opt, varargin{i}), error('EMGPN:InvalidInput', 'Unknown option %s.', varargin{i}); end
    Opt.(varargin{i}) = varargin{i + 1};
end
if ~(isnumeric(Opt.Precision) && isreal(Opt.Precision) && isscalar(Opt.Precision) ...
        && isfinite(Opt.Precision) && Opt.Precision >= 0 && Opt.Precision <= 17 && Opt.Precision == round(Opt.Precision))
    error('EMGPN:InvalidInput', 'Precision must be an integer in 0..17.');
end
if ~isstruct(Dataset) || ~isscalar(Dataset) || ~isfield(Dataset, 'Xdata')
    error('EMGPN:InvalidInput', 'Dataset must be a scalar struct with Xdata.');
end
Xdata = Dataset.Xdata;
if isnumeric(Xdata), Xdata = {Xdata}; end
if ~iscell(Xdata) || isempty(Xdata)
    error('EMGPN:InvalidInput', 'Xdata must be a nonempty cell of numeric matrices.');
end
[NumRoi, NumSubj] = size(Xdata{1});
M = numel(Xdata);
for m = 1:M
    if ~isnumeric(Xdata{m}) || ~isreal(Xdata{m}) || ~ismatrix(Xdata{m}) ...
            || ~isequal(size(Xdata{m}), [NumRoi, NumSubj]) || any(isinf(Xdata{m}(:)))
        error('EMGPN:InvalidInput', 'Every modality must be a same-sized real matrix, with NaN for missing values.');
    end
end
if isfield(Dataset, 'ModalityName') && numel(Dataset.ModalityName) == M
    ModalityName = Dataset.ModalityName;
else
    ModalityName = arrayfun(@(m) sprintf('Modality%d', m), 1:M, 'UniformOutput', false);
end
if isfield(Dataset, 'RoiName') && numel(Dataset.RoiName) == NumRoi
    RoiName = Dataset.RoiName;
else
    RoiName = arrayfun(@(q) sprintf('ROI%03d', q), (1:NumRoi)', 'UniformOutput', false);
end
if isfield(Dataset, 'SubjectID') && numel(Dataset.SubjectID) == NumSubj
    SubjectID = Dataset.SubjectID(:);
else
    SubjectID = arrayfun(@(j) sprintf('P%04d', j), (1:NumSubj)', 'UniformOutput', false);
end
CheckNames(ModalityName, M, 'ModalityName');
CheckNames(RoiName, NumRoi, 'RoiName');
CheckNames(SubjectID, NumSubj, 'SubjectID');

Header = {Opt.IdColumn};
Column = {cellfun(@Quote, SubjectID, 'UniformOutput', false)};
if isfield(Dataset, 'Ydata') && ~isempty(Dataset.Ydata)
    y = Dataset.Ydata(:);
    if ~isfield(Dataset, 'ClassName') || ~isnumeric(y) || ~isreal(y) || numel(y) ~= NumSubj ...
            || any(~isfinite(y) | y < 1 | y > numel(Dataset.ClassName) | y ~= round(y))
        error('EMGPN:InvalidInput', 'Ydata must contain one valid class index per participant.');
    end
    CheckNames(Dataset.ClassName, numel(Dataset.ClassName), 'ClassName');
    Header{end + 1} = Opt.LabelColumn;
    Column{end + 1} = cellfun(@Quote, Dataset.ClassName(Dataset.Ydata(:)), 'UniformOutput', false);
    Column{end} = Column{end}(:);
end
if isfield(Dataset, 'Covariate') && ~isempty(Dataset.Covariate)
    if numel(Dataset.Covariate.Name) ~= numel(Dataset.Covariate.Value)
        error('EMGPN:InvalidInput', 'Covariate Name and Value counts must match.');
    end
    for i = 1:numel(Dataset.Covariate.Name)
        if numel(Dataset.Covariate.Value{i}) ~= NumSubj
            error('EMGPN:InvalidInput', 'Every covariate must have one value per participant.');
        end
        Header{end + 1} = Dataset.Covariate.Name{i};
        Column{end + 1} = FormatColumn(Dataset.Covariate.Value{i}, Opt.Precision);
    end
end
CheckNames(Header, numel(Header), 'CSV header');
Prefix = strjoin(cellfun(@Quote, Header, 'UniformOutput', false), ',');
for m = 1:M
    Name = strcat(ModalityName{m}, '_', RoiName(:)');
    Prefix = [Prefix, ',', strjoin(cellfun(@Quote, Name, 'UniformOutput', false), ',')];
end
X = zeros(M * NumRoi, NumSubj);
for m = 1:M
    X((m - 1) * NumRoi + (1:NumRoi), :) = double(Xdata{m});
end
Format = sprintf('%%.%df,', Opt.Precision);

Parent = fileparts(File);
if isempty(Parent), Parent = pwd; end
TempFile = [tempname(Parent), '.csv'];
TempCleanup = onCleanup(@() DeleteTemporary(TempFile));
[Fid, Msg] = fopen(TempFile, 'w', 'n', 'UTF-8');
if Fid < 0, error('EMGPN:FileError', 'Cannot open %s for writing: %s', File, Msg); end
Cleanup = onCleanup(@() fclose(Fid));
fprintf(Fid, '%s\n', Prefix);
for j = 1:NumSubj
    Lead = strjoin(cellfun(@(C) C{j}, Column, 'UniformOutput', false), ',');
    Body = strrep(sprintf(Format, X(:, j)), 'NaN', '');
    fprintf(Fid, '%s,%s\n', Lead, Body(1:end - 1));
end
clear Cleanup
[Ok, Msg] = movefile(TempFile, File, 'f');
if ~Ok, error('EMGPN:FileError', 'Cannot commit CSV to %s: %s', File, Msg); end
end

function Text = FormatColumn(Value, Precision)
if isnumeric(Value) || islogical(Value)
    Value = double(Value(:));
    Finite = Value(isfinite(Value));
    if all(Finite == round(Finite))
        Text = arrayfun(@(v) sprintf('%d', v), Value, 'UniformOutput', false);
    else
        Text = arrayfun(@(v) sprintf(sprintf('%%.%df', Precision), v), Value, 'UniformOutput', false);
    end
    Text(isnan(Value)) = {''};
else
    Text = cellfun(@Quote, cellstr(Value(:)), 'UniformOutput', false);
end
end

function s = Quote(s)
if any(s == ',') || any(s == '"') || any(s == sprintf('\n')) || any(s == sprintf('\r'))
    s = ['"', strrep(s, '"', '""'), '"'];
end
end

function CheckNames(Name, Count, Field)
if numel(Name) ~= Count || numel(unique(Name)) ~= Count || any(cellfun(@(s) isempty(strtrim(s)), Name))
    error('EMGPN:InvalidInput', '%s must contain %d nonempty unique names.', Field, Count);
end
end

function DeleteTemporary(File)
if exist(File, 'file') == 2, delete(File); end
end
