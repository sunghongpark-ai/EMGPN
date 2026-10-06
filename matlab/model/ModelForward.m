function Out = ModelForward(model, X)

M = model.NumModal;
H = cell(1, M);
for m = 1:M
    switch model.Propagation
        case 'gpn'
            H{m} = PropSolve(model, m, model.Phi(:, m) .* X{m});
        case 'gcn'
            H{m} = model.Agcn{m} * X{m};
        otherwise
            H{m} = X{m};
    end
end
if strcmp(model.Integration, 'concat')
    Z = vertcat(H{:});
else
    Z = model.LambdaProb(:, 1) .* H{1};
    for m = 2:M
        Z = Z + model.LambdaProb(:, m) .* H{m};
    end
end
S = model.Theta' * Z;
if model.UseBias
    S = S + model.Bias;
end
Out = LogitSoftmax(S);
Out.H = H;
Out.Z = Z;
end
