function model = BackwardPropagate(model)

r = model.NumRoi;
M = model.NumModal;
delta = model.RegCoeff;
IsGpn = strcmp(model.Propagation, 'gpn');
IsSoftmax = strcmp(model.Integration, 'softmax');
IsConcat = strcmp(model.Integration, 'concat');

E = (model.Train.P - model.Ytrain) / model.NumTrain;
GradTheta = zeros(size(model.Theta));
Omega = zeros(r, M);
GradPhi = zeros(r, M);
for m = 1:M
    W = ClassifierWeight(model, m);
    if IsGpn
        XE = model.Xtrain{m} * E';
        HE = PropSolve(model, m, model.Phi(:, m) .* XE);
        switch model.GradMode
            case 'exact'
                g = sum(model.Cache.Y{m} .* (XE - HE), 2);
            case 'published'
                g = sum(W .* PropSolve(model, m, XE - HE), 2);
            otherwise
                V = PropSolve(model, m, XE);
                g = sum(W .* V, 2) - model.Phi(:, m) .* sum(W .* PropSolve(model, m, V), 2);
        end
        GradPhi(:, m) = g + 2 * delta * model.Phi(:, m);
    else
        HE = model.Hfix.Train{m} * E';
    end
    if IsConcat
        GradTheta((m - 1) * r + (1:r), :) = HE;
    else
        GradTheta = GradTheta + model.LambdaProb(:, m) .* HE;
        if IsSoftmax
            Omega(:, m) = sum(model.Theta .* HE, 2);
        end
    end
end
Grad.Theta = GradTheta + 2 * delta * model.Theta;
if IsSoftmax
    Lp = model.LambdaProb;
    Grad.Lambda = Lp .* (Omega - sum(Lp .* Omega, 2)) + 2 * delta * model.Lambda;
end
if IsGpn
    Grad.Phi = GradPhi;
end
if model.UseBias
    Grad.Bias = sum(E, 2);
end
G = cell(numel(model.ParamName), 1);
for i = 1:numel(model.ParamName)
    G{i} = Grad.(model.ParamName{i})(:);
end
model.Gradient = vertcat(G{:});
end
