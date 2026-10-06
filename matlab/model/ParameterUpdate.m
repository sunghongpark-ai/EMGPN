function model = ParameterUpdate(model)

g = model.Gradient;
if ~isequal(size(g), size(model.WeightParam)) || ~isreal(g) || any(~isfinite(g))
    error('EMGPN:NonFiniteGradient', 'Gradient must match WeightParam and contain finite real values.');
end
switch model.Optimizer
    case 'adam'
        a = model.AdamParam;
        a.t = a.t + 1;
        a.m = a.beta1 * a.m + (1 - a.beta1) * g;
        a.v = a.beta2 * a.v + (1 - a.beta2) * (g .^ 2);
        mhat = a.m / (1 - a.beta1 ^ a.t);
        vhat = a.v / (1 - a.beta2 ^ a.t);
        model.WeightParam = model.WeightParam - a.alpha * mhat ./ (sqrt(vhat) + a.epsilon);
        model.AdamParam = a;
    otherwise
        model.WeightParam = model.WeightParam - model.LearnRate * g;
end
if any(~isfinite(model.WeightParam)) || (strcmp(model.Optimizer, 'adam') ...
        && (any(~isfinite(model.AdamParam.m)) || any(~isfinite(model.AdamParam.v))))
    error('EMGPN:NonFiniteUpdate', 'The optimizer update produced non-finite parameters or moments.');
end
if ~isempty(model.PhiIndex)
    Phi = model.WeightParam(model.PhiIndex);
    Low = Phi < model.PhiMin;
    if any(Low)
        Phi(Low) = model.PhiMin;
        model.WeightParam(model.PhiIndex) = Phi;
        model.NumProject = model.NumProject + nnz(Low);
    end
end
end
