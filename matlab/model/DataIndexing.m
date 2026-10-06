function model = DataIndexing(model, varargin)

switch numel(varargin)
    case 1
        IdxModel = varargin{1};
        if ~(isscalar(IdxModel) && IdxModel >= 1 && IdxModel <= model.NumModel && IdxModel == round(IdxModel))
            error('EMGPN:InvalidInput', 'IdxModel must be an integer in 1..%d.', model.NumModel);
        end
        Row = model.CVlist(IdxModel, :);
        Fold = model.CVindex(Row(1), :);
        IdxTest = find(Fold == Row(2));
        if model.UseValid
            IdxValid = find(Fold == Row(3));
            IdxTrain = find(Fold ~= Row(2) & Fold ~= Row(3));
        else
            IdxValid = zeros(1, 0);
            IdxTrain = find(Fold ~= Row(2));
        end
        model.IdxModel = IdxModel;
        model.IdxIter = Row(1);
        model.IdxFold = Row(2);
    case 3
        IdxTrain = varargin{1}; IdxValid = varargin{2}; IdxTest = varargin{3};
        model.IdxModel = 0;
        model.IdxIter = 0;
        model.IdxFold = 0;
    otherwise
        error('EMGPN:InvalidInput', 'Use DataIndexing(model, IdxModel) or DataIndexing(model, IdxTrain, IdxValid, IdxTest).');
end

IdxTrain = CheckIndex(IdxTrain, model.NumSubj, 'IdxTrain');
IdxValid = CheckIndex(IdxValid, model.NumSubj, 'IdxValid');
IdxTest = CheckIndex(IdxTest, model.NumSubj, 'IdxTest');
if isempty(IdxTrain)
    error('EMGPN:InvalidInput', 'The training set is empty.');
end
if ~isempty(intersect(IdxTrain, IdxValid)) || ~isempty(intersect(IdxTrain, IdxTest)) ...
        || ~isempty(intersect(IdxValid, IdxTest))
    error('EMGPN:InvalidInput', 'Training, validation, and test sets must be disjoint.');
end

model.IdxTrain = IdxTrain;  model.NumTrain = numel(IdxTrain);
model.IdxValid = IdxValid;  model.NumValid = numel(IdxValid);
model.IdxTest = IdxTest;    model.NumTest = numel(IdxTest);
model.ytrain = model.Ydata(IdxTrain);
model.yvalid = model.Ydata(IdxValid);
model.ytest = model.Ydata(IdxTest);
model.Ytrain = OneHot(model.ytrain, model.NumClass);
model.Yvalid = OneHot(model.yvalid, model.NumClass);
model.Ytest = OneHot(model.ytest, model.NumClass);
model.RandSeed = mod(model.Seed + model.IdxIter, 2^32);
end

function Idx = CheckIndex(Idx, NumSubj, Name)
if islogical(Idx)
    if numel(Idx) ~= NumSubj
        error('EMGPN:InvalidInput', 'Logical %s must have %d elements.', Name, NumSubj);
    end
    Idx = find(Idx);
end
if ~isnumeric(Idx) || ~isreal(Idx) || (~isempty(Idx) && ~isvector(Idx))
    error('EMGPN:InvalidInput', '%s must be a real numeric index vector or logical mask.', Name);
end
Idx = double(Idx(:)');
if any(~isfinite(Idx) | Idx < 1 | Idx > NumSubj | Idx ~= round(Idx)) || numel(unique(Idx)) ~= numel(Idx)
    error('EMGPN:InvalidInput', '%s must contain distinct participant indices in 1..%d.', Name, NumSubj);
end
end

function Y = OneHot(y, NumClass)
Y = zeros(NumClass, numel(y));
Y(sub2ind(size(Y), y, 1:numel(y))) = 1;
end
