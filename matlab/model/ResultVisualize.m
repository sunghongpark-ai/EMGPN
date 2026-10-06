function Fig = ResultVisualize(Result, Folder, Visible)

if nargin < 2, Folder = ''; end
if nargin < 3 || isempty(Visible), Visible = 'on'; end
Fig = GraphicsArray(0);
Fig(1) = PlotPerformance(Result, Visible);
if ~isempty(Result.Explain)
    Fig(2) = PlotModality(Result.Explain, Visible);
    Fig(3) = PlotRiskEffect(Result.Explain, Visible);
    Fig(4) = PlotKeyRegion(Result.Explain, Visible);
end
if ~isempty(Folder)
    if exist(Folder, 'dir') ~= 7, mkdir(Folder); end
    Name = {'Fig_Performance', 'Fig_ModalityImportance', 'Fig_RiskEffect', 'Fig_KeyRegion'};
    for i = 1:numel(Fig)
        print(Fig(i), fullfile(Folder, [Name{i}, '.png']), '-dpng', '-r150');
    end
end
end

function h = GraphicsArray(n)
if exist('gobjects') >= 2
    h = gobjects(1, n);
else
    h = zeros(1, n);
end
end

function Color = ModalityColor(M)
Base = [0.91 0.51 0.37; 0.23 0.61 0.50; 0.36 0.48 0.56];
if M <= 3
    Color = Base(1:M, :);
else
    Color = [Base; lines(M - 3)];
end
end

function F = PlotPerformance(Result, Visible)
F = figure('Visible', Visible, 'Color', 'w', 'Position', [80 80 1250 380], 'Name', 'Performance');
y = Result.Ydata;
NumClass = numel(Result.ClassName);
Score = cell2mat(cellfun(@(P) P(:)', Result.RiskTest(:)', 'UniformOutput', false));
Label = zeros(NumClass, numel(y));
Label(sub2ind(size(Label), y, 1:numel(y))) = 1;
Label = repmat(Label(:)', 1, numel(Result.RiskTest));
[Fpr, Tpr, Rec, Prec] = Curve(Score, Label);
Idx = @(Name) find(strcmp(Result.MetricName, Name));

subplot(1, 3, 1);
plot(Fpr, Tpr, 'Color', [0.2 0.2 0.2], 'LineWidth', 1.5); hold on
plot([0 1], [0 1], ':', 'Color', [0.6 0.6 0.6]); hold off
axis([0 1 0 1]); axis square; box on
xlabel('False positive rate'); ylabel('True positive rate');
title(sprintf('ROC (micro)  AUROC_{macro} = %.4f', Result.MetricMean(Idx('AUROC'))));

subplot(1, 3, 2);
plot(Rec, Prec, 'Color', [0.2 0.2 0.2], 'LineWidth', 1.5); hold on
plot([0 1], [1 1] / NumClass, ':', 'Color', [0.6 0.6 0.6]); hold off
axis([0 1 0 1]); axis square; box on
xlabel('Recall'); ylabel('Precision');
title(sprintf('PR (micro)  AUPRC_{macro} = %.4f', Result.MetricMean(Idx('AUPRC'))));

subplot(1, 3, 3);
Main = {'AUROC', 'AUPRC', 'Accuracy', 'Precision', 'Recall', 'F1'};
K = cellfun(Idx, Main);
bar(1:6, Result.MetricMean(K), 0.6, 'FaceColor', [0.35 0.35 0.35]); hold on
errorbar(1:6, Result.MetricMean(K), Result.MetricStd(K), 'k.'); hold off
set(gca, 'XTick', 1:6, 'XTickLabel', {'AUROC', 'AUPRC', 'Acc.', 'Prec.', 'Rec.', 'F1'}); ylim([0 1]); box on
title('Mean (+/- SD) over repetitions');
end

function [Fpr, Tpr, Rec, Prec] = Curve(Score, Label)
[Sorted, Order] = sort(Score, 'descend');
Hit = Label(Order) > 0;
TP = cumsum(Hit); FP = cumsum(~Hit);
Last = [Sorted(1:end - 1) ~= Sorted(2:end), true];
TP = TP(Last); FP = FP(Last);
Tpr = [0, TP / max(TP(end), 1)];
Fpr = [0, FP / max(FP(end), 1)];
Rec = [0, TP / max(TP(end), 1)];
Prec = [1, TP ./ (TP + FP)];
end

function F = PlotModality(Ex, Visible)
F = figure('Visible', Visible, 'Color', 'w', 'Position', [80 80 1250 380], 'Name', 'ModalityImportance');
Lp = Ex.ModalityImportance;
[NumRoi, M] = size(Lp);
Color = ModalityColor(M);

subplot(1, 3, 1); hold on
for m = 1:M
    BoxPlot(m, Lp(:, m), Color(m, :));
end
hold off; box on
set(gca, 'XTick', 1:M, 'XTickLabel', Ex.ModalityName); xlim([0.4, M + 0.6]);
ylabel('Modality importance'); title('Overall comparison');

subplot(1, 3, 2);
[~, Order] = sort(Lp(:, end), 'ascend');
h = area(1:NumRoi, Lp(Order, :));
for m = 1:M, set(h(m), 'FaceColor', Color(m, :), 'EdgeColor', 'none'); end
axis([1 NumRoi 0 1]); box on
xlabel(sprintf('ROIs sorted by importance of %s', Ex.ModalityName{end})); ylabel('Cumulative importance');
title('Individual comparison');

subplot(1, 3, 3);
Count = Ex.DominantCount;
Keep = find(Count > 0);
if isempty(Keep)
    axis off; text(0.5, 0.5, 'No dominant modality', 'HorizontalAlignment', 'center');
else
    Label = arrayfun(@(m) sprintf('%s: %d (%.1f%%)', Ex.ModalityName{m}, Count(m), 100 * Count(m) / sum(Count)), ...
        Keep, 'UniformOutput', false);
    hp = pie(Count(Keep), Label);
    Patch = hp(strcmp(get(hp, 'Type'), 'patch'));
    for i = 1:min(numel(Patch), numel(Keep)), set(Patch(i), 'FaceColor', Color(Keep(i), :)); end
end
title('# Significant ROIs');
end

function BoxPlot(x, v, Color)
v = sort(v(:));
Q = Quantile(v, [0.25 0.5 0.75]);
Iqr = Q(3) - Q(1);
Lo = min(v(v >= Q(1) - 1.5 * Iqr)); Hi = max(v(v <= Q(3) + 1.5 * Iqr));
w = 0.25;
patch(x + [-w w w -w], [Q(1) Q(1) Q(3) Q(3)], Color, 'FaceAlpha', 0.6, 'EdgeColor', Color * 0.7);
plot(x + [-w w], [Q(2) Q(2)], 'Color', Color * 0.6, 'LineWidth', 1.5);
plot([x x], [Lo Q(1)], 'Color', Color * 0.7); plot([x x], [Q(3) Hi], 'Color', Color * 0.7);
plot(x + [-w w] / 2, [Lo Lo], 'Color', Color * 0.7); plot(x + [-w w] / 2, [Hi Hi], 'Color', Color * 0.7);
Out = v(v < Lo | v > Hi);
plot(repmat(x, size(Out)), Out, 'o', 'Color', Color, 'MarkerSize', 3);
plot(x, mean(v), 'x', 'Color', Color * 0.6, 'MarkerSize', 8, 'LineWidth', 1.5);
end

function Q = Quantile(Sorted, p)
n = numel(Sorted);
Pos = (n - 1) * p + 1;
Lo = floor(Pos); Hi = min(Lo + 1, n);
Q = Sorted(Lo)' + (Pos - Lo) .* (Sorted(Hi)' - Sorted(Lo)');
end

function F = PlotRiskEffect(Ex, Visible)
NumClass = numel(Ex.ClassName);
F = figure('Visible', Visible, 'Color', 'w', 'Position', [80 80 260 * NumClass 300], 'Name', 'RiskEffect');
Bound = max(abs(Ex.RiskEffect(:)));
if ~(Bound > 0), Bound = 1; end
NumBin = 40;
Edge = linspace(-Bound, Bound, NumBin + 1);
Center = (Edge(1:end - 1) + Edge(2:end)) / 2;
for k = 1:NumClass
    v = Ex.RiskEffect(:, k);
    Bin = min(floor((v + Bound) / (2 * Bound) * NumBin) + 1, NumBin);
    Count = accumarray(Bin, 1, [NumBin, 1])';
    subplot(1, NumClass, k); hold on
    Neg = Center < 0;
    bar(Center(Neg), Count(Neg), 1, 'FaceColor', [0.27 0.55 0.85], 'EdgeColor', 'none');
    bar(Center(~Neg), Count(~Neg), 1, 'FaceColor', [0.90 0.25 0.36], 'EdgeColor', 'none');
    plot([0 0], [0 max(Count) + 1], 'k:'); hold off; box on
    xlim([-Bound Bound]); if k == 1, ylabel('Frequency'); end
    xlabel(sprintf('\\Theta  (%d neg / %d pos)', Ex.RiskNegativeCount(k), Ex.RiskPositiveCount(k)));
    title(sprintf('%s (avg %+.4f)', Ex.ClassName{k}, Ex.RiskEffectMean(k)));
end
end

function F = PlotKeyRegion(Ex, Visible)
NumClass = numel(Ex.ClassName);
F = figure('Visible', Visible, 'Color', 'w', 'Position', [80 80 1300 560], 'Name', 'KeyRegion');
Col = max(3, NumClass);
FIn = Ex.FeatureImportanceNorm; PIn = Ex.PermutationImportanceNorm; Key = Ex.IsKey;

subplot(2, Col, 1); hold on
if any(~Key), scatter(FIn(~Key), PIn(~Key), 16, [0.75 0.75 0.75], 'filled'); end
if any(Key), scatter(FIn(Key), PIn(Key), 24, [0.25 0.25 0.25], 'filled'); end
plot(mean(FIn) * [1 1], [0 1], 'k--'); plot([0 1], mean(PIn) * [1 1], 'k--'); hold off
axis([0 1 0 1]); axis square; box on
xlabel('Feature importance'); ylabel('Permutation importance');
title(sprintf('%d key ROIs', nnz(Key)));

subplot(2, Col, 2);
bar([1 2], [Ex.GroupLogPKey, Ex.GroupLogPRest], 0.6, 'FaceColor', [0.3 0.3 0.3]);
set(gca, 'XTick', [1 2], 'XTickLabel', {'Key', 'Rest'}); box on
ylabel('-log_{10} P'); title('Group comparison');

subplot(2, Col, 3);
if isempty(Ex.KeyRoi)
    axis off; text(0.5, 0.5, 'No key ROI', 'HorizontalAlignment', 'center');
else
    [~, Order] = sort(Ex.CombinedImportance(Ex.KeyRoi), 'ascend');
    Top = Ex.KeyRoi(Order(max(1, end - 14):end));
    barh(1:numel(Top), Ex.CombinedImportance(Top), 'FaceColor', [0.3 0.3 0.3]);
    set(gca, 'YTick', 1:numel(Top), 'YTickLabel', Ex.RoiName(Top), 'TickLabelInterpreter', 'none', 'FontSize', 7); box on
    xlabel('Combined importance'); title('Top key ROIs');
end

Color = lines(NumClass);
for k = 1:NumClass
    subplot(2, Col, Col + k); hold on
    x = Ex.CombinedImportance(Ex.KeyRoi); v = Ex.ProbChange(Ex.KeyRoi, k); Up = v > 0;
    if any(Up), scatter(x(Up), v(Up), 24, Color(k, :), 'filled'); end
    if any(~Up), scatter(x(~Up), v(~Up), 24, Color(k, :)); end
    plot([0 1], [0 0], 'k:'); hold off; box on; xlim([0 1]);
    xlabel('Combined importance'); if k == 1, ylabel('Prob. change (%)'); end
    title(sprintf('%s: %d ROIs', Ex.ClassName{k}, nnz(Up)));
end
end
