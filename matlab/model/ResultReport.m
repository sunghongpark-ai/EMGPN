function ResultReport(Result, NumTop)

if nargin < 2, NumTop = 10; end
fprintf('\n==================== EMGPN cross-validation ====================\n');
fprintf('%s | %s\n', Result.Info.Version, Result.Info.Software);
fprintf('Repetitions: %d | models: %d | mean best epoch: %.1f\n', ...
    size(Result.Metric, 1), Result.Info.NumModel, Result.Info.MeanBestEpoch);
fprintf('\nPerformance (mean +/- SD over repetitions)\n');
for k = 1:numel(Result.MetricName)
    fprintf('  %-17s %.4f (+/- %.4f)\n', Result.MetricName{k}, Result.MetricMean(k), Result.MetricStd(k));
end
Ex = Result.Explain;
if isempty(Ex)
    fprintf('================================================================\n');
    return
end
NumModal = numel(Ex.ModalityName);
fprintf('\nModality importance (whole-brain mean; #ROIs dominated)\n');
for m = 1:NumModal
    fprintf('  %-8s %.4f   %4d ROIs (%.1f%%)\n', Ex.ModalityName{m}, Ex.ModalityImportanceMean(m), ...
        Ex.DominantCount(m), 100 * Ex.DominantCount(m) / max(sum(Ex.DominantCount), 1));
end
fprintf('\nRisk effects (mean; #positive / #negative ROIs)\n');
for k = 1:numel(Ex.ClassName)
    fprintf('  %-8s %+.4f   %4d / %4d\n', Ex.ClassName{k}, Ex.RiskEffectMean(k), ...
        Ex.RiskPositiveCount(k), Ex.RiskNegativeCount(k));
end
fprintf('\nKey regions: %d ROIs (mean -log10 P: key %.2f vs rest %.2f)\n', ...
    numel(Ex.KeyRoi), Ex.GroupLogPKey, Ex.GroupLogPRest);
for i = 1:min(NumTop, numel(Ex.KeyRoi))
    q = Ex.KeyRoi(i);
    fprintf('  %2d. %-24s combined %.3f\n', i, Ex.RoiName{q}, Ex.CombinedImportance(q));
end
fprintf('Subtype-specific key regions (increase the subtype probability)\n');
for k = 1:numel(Ex.ClassName)
    fprintf('  %-8s %3d ROIs\n', Ex.ClassName{k}, numel(Ex.SubtypeKeyRoi{k}));
end
fprintf('================================================================\n');
end
