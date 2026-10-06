function Result = ResultSummary(ModelInit, Fold)

y = ModelInit.Ydata;
NumSubj = ModelInit.NumSubj;
NumClass = ModelInit.NumClass;
NumIter = ModelInit.NumIter;
NumFold = numel(Fold);

RiskTest = repmat({NaN(NumClass, NumSubj)}, NumIter, 1);
NumFeat = size(Fold{1}.Ztest, 1);
Zsum = zeros(NumFeat, NumSubj);
Zcount = zeros(1, NumSubj);
for f = 1:NumFold
    F = Fold{f};
    RiskTest{F.IdxIter}(:, F.IdxTest) = F.Ptest;
    Zsum(:, F.IdxTest) = Zsum(:, F.IdxTest) + double(F.Ztest);
    Zcount(F.IdxTest) = Zcount(F.IdxTest) + 1;
    Fold{f} = rmfield(F, 'Ztest');
end
Result.RiskTest = RiskTest;
RiskSum = zeros(NumClass, NumSubj);
for i = 1:NumIter, RiskSum = RiskSum + RiskTest{i}; end
Result.RiskMean = RiskSum / NumIter;
Result.Zoof = Zsum ./ max(Zcount, 1);
Result.Ydata = y;

Result.MetricName = {'AUROC', 'AUPRC', 'Accuracy', 'Precision', 'Recall', 'F1', ...
    'MicroAUROC', 'MicroAUPRC', 'BalancedAccuracy', 'LogLoss', 'PrecisionOvR', 'RecallOvR', 'F1OvR'};
Result.Metric = NaN(NumIter, numel(Result.MetricName));
for i = 1:NumIter
    if any(isnan(RiskTest{i}(:)))
        warning('EMGPN:IncompleteIteration', 'Repetition %d lacks out-of-fold predictions.', i);
        continue
    end
    Perf = PerformMeasure(RiskTest{i}, y);
    PerfOvr = PerformMeasure(RiskTest{i}, y, 'Decision', 'ovr');
    Perf.PrecisionOvR = PerfOvr.Precision;
    Perf.RecallOvR = PerfOvr.Recall;
    Perf.F1OvR = PerfOvr.F1;
    for k = 1:numel(Result.MetricName)
        Result.Metric(i, k) = Perf.(Result.MetricName{k});
    end
end
Done = all(~isnan(Result.Metric), 2);
Result.MetricMean = mean(Result.Metric(Done, :), 1);
if nnz(Done) > 1
    Result.MetricStd = std(Result.Metric(Done, :), 0, 1);
else
    Result.MetricStd = zeros(1, numel(Result.MetricName));
end

if ~strcmp(ModelInit.Integration, 'concat')
    Part.ModalityName = ModelInit.ModalityName;
    Part.ClassName = ModelInit.ClassName;
    Part.RoiName = ModelInit.RoiName;
    Part.ModalityImportance = FoldMean(Fold, 'LambdaProb');
    Part.Steadiness = FoldMean(Fold, 'Phi');
    Part.RiskEffect = FoldMean(Fold, 'Theta');
    Part.FeatureImportance = FoldMean(Fold, 'FeatureImportance');
    Part.PermutationImportance = FoldMean(Fold, 'PermutationImportance');
    Part.ProbChange = FoldMean(Fold, 'ProbChange');
    Part.ProbChangeAbs = FoldMean(Fold, 'ProbChangeAbs');
    Part.GroupPValue = GroupTest(Result.Zoof, y);
    Result.Explain = ExplainSummary(Part);
else
    Result.Explain = [];
end

Result.Fold = Fold;
Result.CVindex = ModelInit.CVindex;
Result.Parameter = ModelInit.Parameter;
Result.ModalityName = ModelInit.ModalityName;
Result.ClassName = ModelInit.ClassName;
Result.RoiName = ModelInit.RoiName;
Result.SubjectID = ModelInit.SubjectID;
if exist('OCTAVE_VERSION', 'builtin') ~= 0
    Software = ['GNU Octave ', OCTAVE_VERSION];
    Stamp = strftime('%Y-%m-%d %H:%M:%S', localtime(time()));
else
    Software = ['MATLAB ', version];
    Stamp = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
end
Result.Info = struct('Method', 'EMGPN (Park et al., Neural Networks 192:107971, 2025)', ...
    'Version', 'EMGPN-MATLAB 2.0.0', 'Software', Software, ...
    'Date', Stamp, 'NumModel', NumFold, ...
    'MeanBestEpoch', mean(cellfun(@(F) F.BestEpoch, Fold)), ...
    'NumPhiProjection', sum(cellfun(@(F) F.NumProject, Fold)));
end

function A = FoldMean(Fold, Name)

if isempty(Fold{1}.(Name))
    A = [];
    return
end
Sum = zeros(size(Fold{1}.(Name)));
Count = zeros(size(Sum));
for f = 1:numel(Fold)
    V = double(Fold{f}.(Name));
    Valid = ~isnan(V);
    Sum(Valid) = Sum(Valid) + V(Valid);
    Count = Count + Valid;
end
A = Sum ./ Count;
A(Count == 0) = NaN;
end
