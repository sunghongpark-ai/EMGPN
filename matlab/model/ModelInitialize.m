function model = ModelInitialize(Dataset, Parameter)

if nargin < 2 || isempty(Parameter), Parameter = struct(); end
if ~isstruct(Dataset) || ~isscalar(Dataset) || ~isfield(Dataset, 'Xdata') || ~isfield(Dataset, 'Ydata')
    error('EMGPN:InvalidInput', 'Dataset must be a struct with fields Xdata and Ydata.');
end
if ~isstruct(Parameter) || ~isscalar(Parameter)
    error('EMGPN:InvalidInput', 'Parameter must be a scalar struct.');
end
Parameter = ResolveParameter(Parameter);

Xdata = Dataset.Xdata;
if isnumeric(Xdata), Xdata = {Xdata}; end
if ~iscell(Xdata) || isempty(Xdata)
    error('EMGPN:InvalidInput', 'Dataset.Xdata must be a non-empty 1-by-M cell of r-by-n matrices.');
end
NumModalTotal = numel(Xdata);
if isfield(Dataset, 'ModalityName') && ~isempty(Dataset.ModalityName)
    ModalityName = cellstr(Dataset.ModalityName);
    ModalityName = ModalityName(:)';
    if numel(ModalityName) ~= NumModalTotal
        error('EMGPN:InvalidInput', 'Dataset.ModalityName must have %d entries.', NumModalTotal);
    end
elseif NumModalTotal == 3
    ModalityName = {'sMRI', 'fMRI', 'PET'};
else
    ModalityName = arrayfun(@(k) sprintf('Modality%d', k), 1:NumModalTotal, 'UniformOutput', false);
end
CheckNames(ModalityName, 'ModalityName');
Modality = Parameter.Modality;
if isempty(Modality)
    Modality = 1:NumModalTotal;
elseif ischar(Modality) || iscell(Modality) || isstring(Modality)
    [Found, Modality] = ismember(cellstr(Modality), ModalityName);
    if ~all(Found)
        error('EMGPN:InvalidParameter', 'Unknown modality name; available: %s.', strjoin(ModalityName, ', '));
    end
end
if ~isnumeric(Modality) || ~isreal(Modality) || ~isvector(Modality)
    error('EMGPN:InvalidParameter', 'Modality must be a vector of indices or modality names.');
end
Modality = double(Modality(:)');
if any(~isfinite(Modality)) || any(Modality < 1) || any(Modality > NumModalTotal) || any(Modality ~= round(Modality)) ...
        || numel(unique(Modality)) ~= numel(Modality)
    error('EMGPN:InvalidParameter', 'Parameter.Modality must select distinct modalities among 1..%d.', NumModalTotal);
end
Modality = sort(Modality);
Parameter.Modality = Modality;
if ~isempty(Parameter.KernelWidth) && ~any(numel(Parameter.KernelWidth) == [1, numel(Modality), NumModalTotal])
    error('EMGPN:InvalidParameter', 'KernelWidth must have 1, %d, or %d elements.', numel(Modality), NumModalTotal);
end
[NumRoi, NumSubj] = size(Xdata{Modality(1)});
Xraw = cell(1, numel(Modality));
for m = 1:numel(Modality)
    X = Xdata{Modality(m)};
    if ~isnumeric(X) || ~isreal(X) || ~ismatrix(X)
        error('EMGPN:InvalidInput', 'Dataset.Xdata{%d} must be a real numeric matrix.', Modality(m));
    end
    if ~isequal(size(X), [NumRoi, NumSubj])
        error('EMGPN:InvalidInput', 'All modalities must be %d-by-%d (ROI x participant).', NumRoi, NumSubj);
    end
    X = double(X);
    if any(isinf(X(:)))
        error('EMGPN:InvalidInput', 'Dataset.Xdata{%d} contains Inf; use NaN for missing values.', Modality(m));
    end
    Xraw{m} = X;
end
if NumRoi < 2, error('EMGPN:InvalidInput', 'At least two ROIs are required.'); end

Ydata = Dataset.Ydata;
if ~(isnumeric(Ydata) || islogical(Ydata)) || ~isreal(Ydata)
    error('EMGPN:InvalidInput', 'Dataset.Ydata must be real numeric labels or one-hot values.');
end
if isempty(Ydata)
    error('EMGPN:InvalidInput', 'Training requires diagnosis labels (Dataset.Ydata).');
end
if ~isvector(Ydata) || numel(Ydata) ~= NumSubj
    if ismatrix(Ydata) && size(Ydata, 2) == NumSubj && size(Ydata, 1) > 1
        if any(abs(sum(double(Ydata), 1) - 1) > 0) || any(Ydata(:) ~= 0 & Ydata(:) ~= 1)
            error('EMGPN:InvalidInput', 'One-hot Dataset.Ydata must have exactly one 1 per column.');
        end
        [~, Ydata] = max(double(Ydata), [], 1);
    else
        error('EMGPN:InvalidInput', 'Dataset.Ydata must hold n = %d labels or a c-by-n one-hot matrix.', NumSubj);
    end
end
Ydata = double(Ydata(:)');
if any(~isfinite(Ydata)) || any(Ydata < 1) || any(Ydata ~= round(Ydata))
    error('EMGPN:InvalidInput', 'Class labels must be positive integers 1..c.');
end
if isfield(Dataset, 'ClassName') && ~isempty(Dataset.ClassName)
    ClassName = cellstr(Dataset.ClassName);
    ClassName = ClassName(:)';
    if max(Ydata) > numel(ClassName)
        error('EMGPN:InvalidInput', 'Labels exceed the number of class names (%d).', numel(ClassName));
    end
else
    ClassName = arrayfun(@(k) sprintf('Class%d', k), 1:max(Ydata), 'UniformOutput', false);
end
CheckNames(ClassName, 'ClassName');
NumClass = numel(ClassName);
if NumClass < 2, error('EMGPN:InvalidInput', 'At least two classes are required.'); end
ClassCount = accumarray(Ydata(:), 1, [NumClass, 1])';

RoiName = NameList(Dataset, 'RoiName', NumRoi, 'ROI%03d');
SubjectID = NameList(Dataset, 'SubjectID', NumSubj, 'P%04d');
Wdata = {};
if isfield(Dataset, 'Wdata') && ~isempty(Dataset.Wdata)
    if ~iscell(Dataset.Wdata) || numel(Dataset.Wdata) ~= NumModalTotal
        error('EMGPN:InvalidInput', 'Dataset.Wdata must be a 1-by-%d cell.', NumModalTotal);
    end
    Wdata = Dataset.Wdata(Modality);
end

if ~isempty(Parameter.CVindex)
    if ~isnumeric(Parameter.CVindex) || ~isreal(Parameter.CVindex) || ~ismatrix(Parameter.CVindex)
        error('EMGPN:InvalidParameter', 'CVindex must be a real numeric matrix.');
    end
    CV = double(Parameter.CVindex);
    if size(CV, 2) ~= NumSubj || any(~isfinite(CV(:))) || any(CV(:) < 1) || any(CV(:) ~= round(CV(:)))
        error('EMGPN:InvalidParameter', 'CVindex must be a NumIter-by-%d matrix of folds 1..K.', NumSubj);
    end
    NumFold = max(CV(:));
    for i = 1:size(CV, 1)
        if numel(unique(CV(i, :))) ~= NumFold
            error('EMGPN:InvalidParameter', 'Row %d of CVindex does not contain all folds 1..%d.', i, NumFold);
        end
    end
    Parameter.NumIter = size(CV, 1);
    Parameter.NumFold = NumFold;
end
if Parameter.UseValid && Parameter.NumFold < 3
    error('EMGPN:InvalidParameter', 'NumFold must be >= 3 when UseValid is true (train/valid/test folds).');
end
if any(ClassCount > 0 & ClassCount < Parameter.NumFold)
    warning('EMGPN:SmallClass', 'Some classes have fewer members than folds; they are absent from some folds.');
end
if isempty(Parameter.CVindex)
    CV = zeros(Parameter.NumIter, NumSubj);
    for IdxIter = 1:Parameter.NumIter
        Stream = RandomStream(mod(Parameter.Seed + IdxIter, 2^32));
        Order = cell(1, NumClass);
        for k = 1:NumClass
            Member = find(Ydata == k);
            [~, Perm] = sort(RandomUniform(Stream, 1, numel(Member)));
            Order{k} = Member(Perm);
        end

        CV(IdxIter, [Order{:}]) = mod(0:NumSubj - 1, Parameter.NumFold) + 1;
    end
end
Fold = (1:Parameter.NumFold)';
CVfold = [Fold, mod(Fold, Parameter.NumFold) + 1];
CVlist = [kron((1:Parameter.NumIter)', ones(Parameter.NumFold, 1)), repmat(CVfold, Parameter.NumIter, 1)];

model = Parameter;
model.Parameter = Parameter;
model.Xraw = Xraw;
model.Ydata = Ydata;
model.Wdata = Wdata;
model.NumRoi = NumRoi;
model.NumSubj = NumSubj;
model.NumClass = NumClass;
model.NumModal = numel(Modality);
model.NumModalTotal = NumModalTotal;
model.ModalityIndex = Modality;
model.ModalityName = ModalityName(Modality);
model.ModalityNameTotal = ModalityName;
model.ClassName = ClassName;
model.ClassCount = ClassCount;
model.RoiName = RoiName;
model.SubjectID = SubjectID;
model.CVindex = CV;
model.CVlist = CVlist;
model.NumModel = size(CVlist, 1);
end

function Name = NameList(Dataset, Field, Count, Pattern)
if isfield(Dataset, Field) && ~isempty(Dataset.(Field))
    Name = cellstr(Dataset.(Field));
    Name = Name(:);
    if numel(Name) ~= Count
        error('EMGPN:InvalidInput', 'Dataset.%s must have %d entries.', Field, Count);
    end
else
    Name = arrayfun(@(k) sprintf(Pattern, k), (1:Count)', 'UniformOutput', false);
end
CheckNames(Name, Field);
end

function CheckNames(Name, Field)
if any(cellfun(@(s) isempty(strtrim(s)), Name)) || numel(unique(Name)) ~= numel(Name)
    error('EMGPN:InvalidInput', 'Dataset.%s entries must be nonempty and unique.', Field);
end
end

function P = ResolveParameter(P)

Default = struct( ...
    'NumIter', 100, 'NumFold', 5, 'UseValid', true, 'Seed', 0, 'CVindex', [], ...
    'MaxEpoch', 500, 'Patience', Inf, 'Optimizer', 'adam', 'LearnRate', 0.005, ...
    'AdamBeta1', 0.9, 'AdamBeta2', 0.999, 'AdamEpsilon', 1e-8, ...
    'RegCoeff', 1e-4, 'InitSteady', 1, 'InitIntegrate', 0, 'InitClassifier', 'glorot', ...
    'GradMode', 'exact', 'PhiMin', 1e-6, 'NumNeighbor', 10, 'KernelWidth', [], ...
    'Normalize', 'zscore-logistic', 'Modality', [], ...
    'Propagation', 'gpn', 'Integration', 'softmax', 'UseBias', false, ...
    'NumPermute', 10, 'PermMetric', 'loss', 'ProbBaseline', 'zero', 'ProbSubset', 'class', ...
    'Verbose', 1);
Unknown = setdiff(fieldnames(P), fieldnames(Default));
if ~isempty(Unknown)
    error('EMGPN:InvalidParameter', 'Unknown Parameter field(s): %s. Valid fields: %s.', ...
        strjoin(Unknown(:)', ', '), strjoin(fieldnames(Default)', ', '));
end
Name = fieldnames(Default);
for i = 1:numel(Name)
    if ~isfield(P, Name{i}), P.(Name{i}) = Default.(Name{i}); end
end
P = orderfields(P, Default);
CheckInteger(P.NumIter, 'NumIter', 1);
CheckInteger(P.NumFold, 'NumFold', 2);
CheckInteger(P.MaxEpoch, 'MaxEpoch', 1);
CheckInteger(P.NumNeighbor, 'NumNeighbor', 1);
CheckInteger(P.NumPermute, 'NumPermute', 1);
CheckInteger(P.Seed, 'Seed', 0);
if P.Seed >= 2^32
    error('EMGPN:InvalidParameter', 'Seed must be an integer in [0, 2^32).');
end
CheckInteger(P.Verbose, 'Verbose', 0);
CheckPositive(P.LearnRate, 'LearnRate');
CheckPositive(P.InitSteady, 'InitSteady');
CheckPositive(P.AdamEpsilon, 'AdamEpsilon');
CheckPositive(P.PhiMin, 'PhiMin');
if ~(isnumeric(P.Patience) && isreal(P.Patience) && isscalar(P.Patience) && P.Patience >= 1 ...
        && (isinf(P.Patience) || P.Patience == round(P.Patience)))
    error('EMGPN:InvalidParameter', 'Patience must be >= 1 (Inf allowed).');
end
if ~(isnumeric(P.RegCoeff) && isscalar(P.RegCoeff) && isfinite(P.RegCoeff) && P.RegCoeff >= 0)
    error('EMGPN:InvalidParameter', 'RegCoeff must be >= 0.');
end
if ~(isnumeric(P.AdamBeta1) && isreal(P.AdamBeta1) && isscalar(P.AdamBeta1) && P.AdamBeta1 >= 0 && P.AdamBeta1 < 1) ...
        || ~(isnumeric(P.AdamBeta2) && isreal(P.AdamBeta2) && isscalar(P.AdamBeta2) && P.AdamBeta2 >= 0 && P.AdamBeta2 < 1)
    error('EMGPN:InvalidParameter', 'AdamBeta1 and AdamBeta2 must lie in [0, 1).');
end
if ~(isnumeric(P.InitIntegrate) && isscalar(P.InitIntegrate) && isfinite(P.InitIntegrate))
    error('EMGPN:InvalidParameter', 'InitIntegrate must be a finite scalar.');
end
if ~isempty(P.KernelWidth) && ~(isnumeric(P.KernelWidth) && all(P.KernelWidth(:) > 0) && all(isfinite(P.KernelWidth(:))))
    error('EMGPN:InvalidParameter', 'KernelWidth must be [] or positive value(s).');
end
P.UseValid = CheckBoolean(P.UseValid, 'UseValid');
P.UseBias = CheckBoolean(P.UseBias, 'UseBias');
if P.InitSteady < P.PhiMin
    error('EMGPN:InvalidParameter', 'InitSteady must be >= PhiMin.');
end
P.Optimizer = CheckOption(P.Optimizer, {'adam', 'gd'}, 'Optimizer');
P.InitClassifier = CheckOption(P.InitClassifier, {'glorot', 'zeros'}, 'InitClassifier');
P.GradMode = CheckOption(P.GradMode, {'exact', 'published', 'legacy'}, 'GradMode');
P.Normalize = CheckOption(P.Normalize, {'zscore-logistic', 'zscore', 'none'}, 'Normalize');
P.Propagation = CheckOption(P.Propagation, {'gpn', 'gcn', 'none'}, 'Propagation');
P.Integration = CheckOption(P.Integration, {'softmax', 'mean', 'concat'}, 'Integration');
P.PermMetric = CheckOption(P.PermMetric, {'loss', 'auroc'}, 'PermMetric');
P.ProbBaseline = CheckOption(P.ProbBaseline, {'zero', 'mean'}, 'ProbBaseline');
P.ProbSubset = CheckOption(P.ProbSubset, {'class', 'all'}, 'ProbSubset');
end

function CheckInteger(Value, Name, Lower)
if ~(isnumeric(Value) && isreal(Value) && isscalar(Value) && isfinite(Value) && Value == round(Value) && Value >= Lower)
    error('EMGPN:InvalidParameter', '%s must be an integer >= %d.', Name, Lower);
end
end

function CheckPositive(Value, Name)
if ~(isnumeric(Value) && isreal(Value) && isscalar(Value) && isfinite(Value) && Value > 0)
    error('EMGPN:InvalidParameter', '%s must be a positive finite scalar.', Name);
end
end

function Value = CheckBoolean(Value, Name)
if ~((isnumeric(Value) || islogical(Value)) && isreal(Value) && isscalar(Value) ...
        && isfinite(Value) && (Value == 0 || Value == 1))
    error('EMGPN:InvalidParameter', '%s must be scalar true/false or 0/1.', Name);
end
Value = logical(Value);
end

function Value = CheckOption(Value, Allowed, Name)
if isstring(Value) && isscalar(Value), Value = char(Value); end
if ~ischar(Value) || ~any(strcmpi(Value, Allowed))
    error('EMGPN:InvalidParameter', '%s must be one of: %s.', Name, strjoin(Allowed, ', '));
end
Value = Allowed{strcmpi(Value, Allowed)};
end
