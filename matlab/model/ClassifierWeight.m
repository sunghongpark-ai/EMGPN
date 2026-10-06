function W = ClassifierWeight(model, m)

if strcmp(model.Integration, 'concat')
    W = model.Theta((m - 1) * model.NumRoi + (1:model.NumRoi), :);
else
    W = model.LambdaProb(:, m) .* model.Theta;
end
end
