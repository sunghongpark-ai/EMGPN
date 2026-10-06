function model = ParamTraining(model)

HasValid = model.NumValid > 0;
BestCrit = Inf;
BestEpoch = 0;
BestWeight = model.WeightParam;
BestAdam = model.AdamParam;
Wait = 0;
NumEpoch = 0;
for IdxEpoch = 1:model.MaxEpoch
    model.IdxEpoch = IdxEpoch;
    model = ParamReshape(model);
    model = ForwardPropagate(model);
    model = LossCalculation(model);
    if ~isfinite(model.ObjTrain(IdxEpoch)) || (HasValid && ~isfinite(model.LossValid(IdxEpoch)))
        warning('EMGPN:NonFiniteLoss', 'Non-finite loss at epoch %d; training stopped.', IdxEpoch);
        break
    end
    NumEpoch = IdxEpoch;
    if HasValid, Crit = model.LossValid(IdxEpoch); else, Crit = -IdxEpoch; end
    if Crit < BestCrit
        BestCrit = Crit;
        BestEpoch = IdxEpoch;
        BestWeight = model.WeightParam;
        BestAdam = model.AdamParam;
        Wait = 0;
    else
        Wait = Wait + 1;
        if Wait >= model.Patience, break, end
    end
    if IdxEpoch < model.MaxEpoch
        model = BackwardPropagate(model);
        model = ParameterUpdate(model);
    end
    if model.Verbose >= 2 && mod(IdxEpoch, 50) == 0
        fprintf('[EMGPN] epoch %4d | train loss %.4f | objective %.4f | valid loss %.4f\n', ...
            IdxEpoch, model.LossTrain(IdxEpoch), model.ObjTrain(IdxEpoch), model.LossValid(IdxEpoch));
    end
end
if BestEpoch == 0
    error('EMGPN:TrainingFailed', 'No finite loss was obtained; check the data and hyperparameters.');
end

model.NumEpoch = NumEpoch;
model.BestEpoch = BestEpoch;
model.LossTrain = model.LossTrain(1:NumEpoch);
model.LossValid = model.LossValid(1:NumEpoch);
model.ObjTrain = model.ObjTrain(1:NumEpoch);
model.WeightParam = BestWeight;
model.AdamParam = BestAdam;
model = ParamReshape(model);
model.Train = ModelForward(model, model.Xtrain);
if HasValid
    model.Valid = ModelForward(model, model.Xvalid);
end
model.ZMean = mean(model.Train.Z, 2);
model.HMean = cellfun(@(h) mean(h, 2), model.Train.H, 'UniformOutput', false);
if model.NumTest > 0
    model.Test = ModelForward(model, model.Xtest);
end
Transient = intersect({'Gradient', 'Cache', 'Hfix', 'IdxEpoch'}, fieldnames(model));
model = rmfield(model, Transient);
if model.Verbose >= 1
    fprintf('[EMGPN] iter %3d fold %d | best epoch %4d / %4d | train loss %.4f | valid loss %.4f\n', ...
        model.IdxIter, model.IdxFold, BestEpoch, NumEpoch, model.LossTrain(BestEpoch), model.LossValid(BestEpoch));
end
end
