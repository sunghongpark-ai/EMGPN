function Explain = ExplainSummary(Part)

Explain = Part;
Lp = Part.ModalityImportance;
M = size(Lp, 2);
Explain.ModalityImportanceMean = mean(Lp, 1);
[Top, Dominant] = max(Lp, [], 2);
Tie = sum(Lp >= Top - 1e-12, 2) > 1;
Dominant(Tie) = NaN;
Explain.DominantModality = Dominant;
Explain.DominantCount = accumarray(Dominant(~Tie), 1, [M, 1])';

Theta = Part.RiskEffect;
Explain.RiskEffectMean = mean(Theta, 1);
Explain.RiskPositiveCount = sum(Theta > 0, 1);
Explain.RiskNegativeCount = sum(Theta < 0, 1);
Explain.RiskPositiveMean = sum(Theta .* (Theta > 0), 1) ./ max(Explain.RiskPositiveCount, 1);
Explain.RiskNegativeMean = sum(Theta .* (Theta < 0), 1) ./ max(Explain.RiskNegativeCount, 1);

Explain.GroupLogP = -log10(max(Part.GroupPValue, realmin));
Explain.GroupLogP(isnan(Part.GroupPValue)) = NaN;
Key = KeyRegion(Part.FeatureImportance, Part.PermutationImportance, Part.ProbChange, Explain.GroupLogP);
KeyField = fieldnames(Key);
for i = 1:numel(KeyField)
    Explain.(KeyField{i}) = Key.(KeyField{i});
end
end
