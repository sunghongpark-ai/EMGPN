function S = SubjectExplain(model, X, j)

model = ParamReshape(model);
NumSubj = size(X{1}, 2);
if ~(isscalar(j) && j >= 1 && j <= NumSubj && j == round(j))
    error('EMGPN:InvalidInput', 'j must be an integer in 1..%d.', NumSubj);
end
Xj = cellfun(@(x) x(:, j), X, 'UniformOutput', false);
Out = ModelForward(model, Xj);
r = model.NumRoi;
M = model.NumModal;
c = model.NumClass;

S.Risk = Out.P;
[~, S.PredictedClass] = max(Out.P);
S.PredictedName = model.ClassName{S.PredictedClass};
S.Logit = Out.S;
S.BaseLogit = model.Theta' * model.ZMean;
if model.UseBias, S.BaseLogit = S.BaseLogit + model.Bias; end
S.Propagated = cell2mat(Out.H);
S.ModalityImportance = model.LambdaProb;
S.ModalityContribution = zeros(r, M, c);
for m = 1:M
    S.ModalityContribution(:, m, :) = reshape(ClassifierWeight(model, m) .* (Out.H{m} - model.HMean{m}), [r, 1, c]);
end
S.Contribution = reshape(sum(S.ModalityContribution, 2), [r, c]);
if strcmp(model.Integration, 'concat')
    S.Integrated = [];
else
    S.Integrated = Out.Z;
end
S.DecompositionError = max(abs(S.BaseLogit + sum(S.Contribution, 1)' - S.Logit));
end
