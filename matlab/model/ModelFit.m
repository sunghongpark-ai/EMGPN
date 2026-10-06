function model = ModelFit(Dataset, Parameter, IdxTrain, IdxValid, IdxTest)

if nargin < 4, IdxValid = []; end
if nargin < 5, IdxTest = []; end
if nargin < 2 || isempty(Parameter), Parameter = struct(); end
Parameter.NumIter = 1;
Parameter.CVindex = [];
Warn = warning('off', 'EMGPN:SmallClass');
try
    model = ModelInitialize(Dataset, Parameter);
catch Err
    warning(Warn);
    rethrow(Err);
end
warning(Warn);
model = DataIndexing(model, IdxTrain, IdxValid, IdxTest);
model = DataNormalize(model);
model = GraphConstruct(model);
model = ParamInitialize(model);
model = ParamTraining(model);
end
