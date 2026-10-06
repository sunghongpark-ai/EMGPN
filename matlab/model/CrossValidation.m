function [Result, ModelInit] = CrossValidation(Dataset, Parameter)

if nargin < 2 || isempty(Parameter), Parameter = struct(); end
ModelInit = ModelInitialize(Dataset, Parameter);
NumModel = ModelInit.NumModel;
Fold = cell(NumModel, 1);
Clock = tic;
HasPool = false;
if exist('OCTAVE_VERSION', 'builtin') == 0 && exist('gcp', 'file') == 2
    HasPool = ~isempty(gcp('nocreate'));
end
if HasPool
    parfor IdxModel = 1:NumModel
        Fold{IdxModel} = TrainFold(ModelInit, IdxModel);
    end
else
    for IdxModel = 1:NumModel
        Fold{IdxModel} = TrainFold(ModelInit, IdxModel);
    end
end
Result = ResultSummary(ModelInit, Fold);
Result.Info.ElapsedSecond = toc(Clock);
end

function F = TrainFold(ModelInit, IdxModel)
Model = DataIndexing(ModelInit, IdxModel);
Model = DataNormalize(Model);
Model = GraphConstruct(Model);
Model = ParamInitialize(Model);
Model = ParamTraining(Model);
F = FoldCompact(Model);
end

function F = FoldCompact(Model)

F.IdxModel = Model.IdxModel;
F.IdxIter = Model.IdxIter;
F.IdxFold = Model.IdxFold;
F.IdxTest = Model.IdxTest;
F.Ptest = Model.Test.P;
F.Ztest = single(Model.Test.Z);
F.LambdaProb = Model.LambdaProb;
F.Theta = Model.Theta;
if strcmp(Model.Propagation, 'gpn'), F.Phi = Model.Phi; else, F.Phi = []; end
if Model.UseBias, F.Bias = Model.Bias; else, F.Bias = []; end
F.Sigma = Model.Sigma;
F.BestEpoch = Model.BestEpoch;
F.NumEpoch = Model.NumEpoch;
F.NumProject = Model.NumProject;
F.LossTrain = Model.LossTrain;
F.LossValid = Model.LossValid;
if strcmp(Model.Integration, 'concat')
    F.FeatureImportance = [];
    F.PermutationImportance = [];
    F.ProbChange = [];
    F.ProbChangeAbs = [];
else
    Ex = ModelExplain(Model, Model.Xtest, Model.ytest, 'GroupTest', false);
    F.FeatureImportance = Ex.FeatureImportance;
    F.PermutationImportance = Ex.PermutationImportance;
    F.ProbChange = Ex.ProbChange;
    F.ProbChangeAbs = Ex.ProbChangeAbs;
end
end
