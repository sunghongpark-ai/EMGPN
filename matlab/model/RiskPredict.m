function [P, Out] = RiskPredict(model, Data)

if ~isfield(model, 'WeightParam') || ~isfield(model, 'NormMean') || ~isfield(model, 'BestEpoch')
    error('EMGPN:InvalidInput', 'RiskPredict requires a trained model (see ModelFit).');
end
model = ParamReshape(model);
Out = ModelForward(model, DataTransform(model, Data));
P = Out.P;
end
