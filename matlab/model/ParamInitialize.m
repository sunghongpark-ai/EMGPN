function model = ParamInitialize(model)

r = model.NumRoi;
M = model.NumModal;
c = model.NumClass;
if strcmp(model.Integration, 'concat'), NumFeat = M * r; else, NumFeat = r; end

Name = {};
Size = zeros(0, 2);
Init = {};
if strcmp(model.Propagation, 'gpn')
    Name{end + 1} = 'Phi';    Size(end + 1, :) = [r, M]; Init{end + 1} = model.InitSteady * ones(r, M);
end
if strcmp(model.Integration, 'softmax')
    Name{end + 1} = 'Lambda'; Size(end + 1, :) = [r, M]; Init{end + 1} = model.InitIntegrate * ones(r, M);
end
switch model.InitClassifier
    case 'glorot'
        Stream = RandomStream(model.RandSeed);
        Theta = (2 * RandomUniform(Stream, NumFeat, c) - 1) * sqrt(6 / (NumFeat + c));
    otherwise
        Theta = zeros(NumFeat, c);
end
Name{end + 1} = 'Theta';      Size(end + 1, :) = [NumFeat, c]; Init{end + 1} = Theta;
if model.UseBias
    Name{end + 1} = 'Bias';   Size(end + 1, :) = [c, 1];       Init{end + 1} = zeros(c, 1);
end

Count = prod(Size, 2);
model.ParamName = Name;
model.ParamSize = Size;
model.ParamStop = cumsum(Count);
model.ParamStart = model.ParamStop - Count + 1;
model.NumFeat = NumFeat;
model.WeightParam = cell2mat(cellfun(@(x) x(:), Init(:), 'UniformOutput', false));
model.NumParam = numel(model.WeightParam);
IdxPhi = find(strcmp(Name, 'Phi'));
if isempty(IdxPhi)
    model.PhiIndex = zeros(0, 1);
else
    model.PhiIndex = (model.ParamStart(IdxPhi):model.ParamStop(IdxPhi))';
end
model.AdamParam = AdamInitialize(model.NumParam, model.LearnRate, model.AdamBeta1, ...
    model.AdamBeta2, model.AdamEpsilon);
model.LossTrain = NaN(model.MaxEpoch, 1);
model.LossValid = NaN(model.MaxEpoch, 1);
model.ObjTrain = NaN(model.MaxEpoch, 1);
model.NumProject = 0;
end
