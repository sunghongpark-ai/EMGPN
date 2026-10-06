function model = ParamReshape(model)

if numel(model.WeightParam) ~= model.NumParam || any(~isfinite(model.WeightParam)) || ~isreal(model.WeightParam)
    error('EMGPN:InvalidInput', 'WeightParam must contain NumParam finite real values.');
end
for i = 1:numel(model.ParamName)
    model.(model.ParamName{i}) = reshape( ...
        model.WeightParam(model.ParamStart(i):model.ParamStop(i)), model.ParamSize(i, :));
end
r = model.NumRoi;
M = model.NumModal;
switch model.Integration
    case 'softmax'
        E = exp(model.Lambda - max(model.Lambda, [], 2));
        model.LambdaProb = E ./ sum(E, 2);
    case 'mean'
        model.LambdaProb = ones(r, M) / M;
    otherwise
        model.LambdaProb = NaN(r, M);
end
if strcmp(model.Propagation, 'gpn')
    model.PropChol = cell(1, M);
    for m = 1:M
        if any(model.Phi(:, m) <= 0)
            error('EMGPN:NotPositiveDefinite', 'Steadiness parameters must be positive (modality %d).', m);
        end
        if issparse(model.Lgraph{m})
            A = model.Lgraph{m} + spdiags(model.Phi(:, m), 0, r, r);
        else
            A = model.Lgraph{m} + diag(model.Phi(:, m));
        end
        [R, p] = chol(A);
        if p ~= 0
            error('EMGPN:NotPositiveDefinite', ...
                'Phi + L of modality %d is not positive definite (min phi = %g); keep PhiMin > 0.', ...
                m, min(model.Phi(:, m)));
        end
        model.PropChol{m} = R;
    end
end
end
