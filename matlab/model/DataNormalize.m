function model = DataNormalize(model)

M = model.NumModal;
model.NormMean = cell(1, M);
model.NormStd = cell(1, M);
for m = 1:M
    Xtr = model.Xraw{m}(:, model.IdxTrain);
    Valid = ~isnan(Xtr);
    Count = sum(Valid, 2);
    X0 = Xtr; X0(~Valid) = 0;
    Mu = sum(X0, 2) ./ max(Count, 1);
    Dev = (X0 - Mu) .* Valid;
    Sd = sqrt(sum(Dev.^2, 2) ./ max(Count - 1, 1));
    if any(~isfinite(Mu(Count > 0))) || any(~isfinite(Sd))
        error('EMGPN:InvalidInput', 'Training normalization overflow in modality %d; rescale finite input values.', m);
    end
    Mu(Count == 0) = NaN;
    Sd(Count <= 1 | ~(Sd > 0)) = 1;
    model.NormMean{m} = Mu;
    model.NormStd{m} = Sd;
end
Xnorm = DataTransform(model, model.Xraw);
model.Xtrain = cell(1, M); model.Xvalid = cell(1, M); model.Xtest = cell(1, M);
for m = 1:M
    model.Xtrain{m} = Xnorm{m}(:, model.IdxTrain);
    model.Xvalid{m} = Xnorm{m}(:, model.IdxValid);
    model.Xtest{m} = Xnorm{m}(:, model.IdxTest);
end
end
