function Xnorm = DataTransform(model, Data)

Xraw = AlignData(model, Data);
NumSubj = size(Xraw{1}, 2);
Xnorm = cell(1, model.NumModal);
for m = 1:model.NumModal
    if ~isnumeric(Xraw{m}) || ~isreal(Xraw{m}) || ~ismatrix(Xraw{m}) || any(isinf(Xraw{m}(:)))
        error('EMGPN:InvalidInput', 'Modality %d must be a real numeric matrix; use NaN for missing values.', m);
    end
    X = double(Xraw{m});
    if ~isequal(size(X), [model.NumRoi, NumSubj])
        error('EMGPN:InvalidInput', 'Modality %d must be %d-by-%d (ROI x participant).', m, model.NumRoi, NumSubj);
    end
    switch model.Normalize
        case {'zscore-logistic', 'zscore'}
            Z = (X - model.NormMean{m}) ./ model.NormStd{m};
        otherwise
            Z = X;
    end
    Z(isnan(Z)) = 0;
    if strcmp(model.Normalize, 'zscore-logistic')
        Z = 1 ./ (1 + exp(-Z));
    end
    Xnorm{m} = Z;
end
end

function X = AlignData(model, Data)
if isstruct(Data)
    if ~isscalar(Data) || ~isfield(Data, 'Xdata')
        error('EMGPN:InvalidInput', 'Data must be a scalar Dataset struct with Xdata.');
    end
    X = Data.Xdata;
    if isnumeric(X), X = {X}; end
    if ~iscell(X) || isempty(X)
        error('EMGPN:InvalidInput', 'Data.Xdata must be a nonempty cell of matrices.');
    end
    if isfield(Data, 'ModalityName') && ~isempty(Data.ModalityName)
        Name = CheckNames(Data.ModalityName, numel(X), 'ModalityName');
        [Found, Loc] = ismember(model.ModalityName, Name);
        if ~all(Found)
            error('EMGPN:InvalidInput', 'Missing modalities: %s.', strjoin(model.ModalityName(~Found), ', '));
        end
        X = X(Loc);
    elseif numel(X) == model.NumModalTotal
        X = X(model.ModalityIndex);
    end
    if isfield(Data, 'RoiName') && ~isempty(Data.RoiName)
        Name = CheckNames(Data.RoiName, size(X{1}, 1), 'RoiName');
        [Found, Loc] = ismember(model.RoiName, Name);
        if ~all(Found)
            error('EMGPN:InvalidInput', '%d ROIs are missing, e.g. %s.', nnz(~Found), model.RoiName{find(~Found, 1)});
        end
        for m = 1:numel(X)
            if ~isnumeric(X{m}) || ~isreal(X{m}) || ~ismatrix(X{m}) || size(X{m}, 1) ~= numel(Name)
                error('EMGPN:InvalidInput', 'Every modality must have one row per RoiName entry.');
            end
            X{m} = X{m}(Loc, :);
        end
    end
else
    X = Data;
    if isnumeric(X), X = {X}; end
end
if ~iscell(X) || isempty(X)
    error('EMGPN:InvalidInput', 'Data must contain a nonempty cell of modality matrices.');
end
if numel(X) == model.NumModalTotal && model.NumModalTotal ~= model.NumModal
    X = X(model.ModalityIndex);
elseif numel(X) ~= model.NumModal
    error('EMGPN:InvalidInput', 'Data must have %d (or %d) modalities.', model.NumModal, model.NumModalTotal);
end
end

function Name = CheckNames(Value, Count, Field)
try
    Name = cellstr(Value);
catch
    error('EMGPN:InvalidInput', '%s must contain text names.', Field);
end
if numel(Name) ~= Count || numel(unique(Name)) ~= Count || any(cellfun(@(s) isempty(strtrim(s)), Name))
    error('EMGPN:InvalidInput', '%s must have %d nonempty unique entries.', Field, Count);
end
end
