function model = LossCalculation(model)

e = model.IdxEpoch;
model.LossTrain(e) = CrossEntropy(model.Train.LogP, model.ytrain);
Reg = 0;
RegName = {'Phi', 'Lambda', 'Theta'};
for i = 1:numel(RegName)
    if any(strcmp(model.ParamName, RegName{i}))
        Reg = Reg + sum(model.(RegName{i})(:) .^ 2);
    end
end
model.ObjTrain(e) = model.LossTrain(e) + model.RegCoeff * Reg;
if model.NumValid > 0
    model.LossValid(e) = CrossEntropy(model.Valid.LogP, model.yvalid);
end
end

function L = CrossEntropy(LogP, y)
L = -mean(LogP(sub2ind(size(LogP), y, 1:numel(y))));
end
