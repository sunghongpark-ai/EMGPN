function model = ForwardPropagate(model)

M = model.NumModal;
c = model.NumClass;
IsGpn = strcmp(model.Propagation, 'gpn');
if ~IsGpn && ~isfield(model, 'Hfix')
    model.Hfix.Train = model.Xtrain;
    model.Hfix.Valid = model.Xvalid;
    if strcmp(model.Propagation, 'gcn')
        for m = 1:M
            model.Hfix.Train{m} = model.Agcn{m} * model.Xtrain{m};
            model.Hfix.Valid{m} = model.Agcn{m} * model.Xvalid{m};
        end
    end
end
Strain = zeros(c, model.NumTrain);
Svalid = zeros(c, model.NumValid);
model.Cache.Y = cell(1, M);
for m = 1:M
    W = ClassifierWeight(model, m);
    if IsGpn
        Y = PropSolve(model, m, W);
        V = model.Phi(:, m) .* Y;
        model.Cache.Y{m} = Y;
        Strain = Strain + V' * model.Xtrain{m};
        if model.NumValid > 0
            Svalid = Svalid + V' * model.Xvalid{m};
        end
    else
        Strain = Strain + W' * model.Hfix.Train{m};
        if model.NumValid > 0
            Svalid = Svalid + W' * model.Hfix.Valid{m};
        end
    end
end
if model.UseBias
    Strain = Strain + model.Bias;
    Svalid = Svalid + model.Bias;
end
model.Train = LogitSoftmax(Strain);
if model.NumValid > 0
    model.Valid = LogitSoftmax(Svalid);
end
end
