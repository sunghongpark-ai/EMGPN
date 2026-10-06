function Result = RunAblation(OutFolder, Parameter)
if nargin < 1 || isempty(OutFolder)
    OutFolder = 'Result';
end
if nargin < 2
    Parameter = struct();
end
DataFile = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'dataset', 'sample.csv');
Dataset = DataRead(DataFile, 'ClassName', {'SCD', 'MCI', 'AD', 'VCI', 'VD'}, ...
    'Modality', {'sMRI', 'fMRI', 'PET'});

Base = struct('NumIter', 10, 'NumFold', 5, 'MaxEpoch', 500, 'LearnRate', 0.005, ...
    'RegCoeff', 1e-4, 'InitSteady', 1, 'NumNeighbor', 10, 'Verbose', 0);
MainMetric = {'AUROC', 'AUPRC', 'Accuracy', 'Precision', 'Recall', 'F1'};
Names = fieldnames(Parameter);
for Index = 1:numel(Names)
    Base.(Names{Index}) = Parameter.(Names{Index});
end
if exist(OutFolder, 'dir') ~= 7, mkdir(OutFolder); end

InitSteady = 10 .^ (-2:2);
RegCoeff = 10 .^ (-5:-1);
SensMean = NaN(numel(InitSteady), numel(RegCoeff));
SensStd = NaN(numel(InitSteady), numel(RegCoeff));
for i = 1:numel(InitSteady)
    for j = 1:numel(RegCoeff)
        Parameter = Base;
        Parameter.InitSteady = InitSteady(i);
        Parameter.RegCoeff = RegCoeff(j);
        Result = CrossValidation(Dataset, Parameter);
        SensMean(i, j) = Result.MetricMean(strcmp(Result.MetricName, 'AUROC'));
        SensStd(i, j) = Result.MetricStd(strcmp(Result.MetricName, 'AUROC'));
        fprintf('InitSteady %g, RegCoeff %g: AUROC %.4f (+/- %.4f)\n', ...
            InitSteady(i), RegCoeff(j), SensMean(i, j), SensStd(i, j));
    end
end
fprintf('\nAUROC by initial steadiness (rows) and regularization (columns)\n%12s', '');
fprintf('%14g', RegCoeff); fprintf('%14s\n', 'Average');
for i = 1:numel(InitSteady)
    fprintf('%12g', InitSteady(i)); fprintf('%14.4f', SensMean(i, :)); fprintf('%14.4f\n', mean(SensMean(i, :)));
end

VariantName = {'EMGPN', 'w/o feature propagation', 'w/o modality integration'};
VariantSetting = {struct(), struct('Propagation', 'gcn'), struct('Integration', 'concat')};
VariantMetric = NaN(numel(VariantName), numel(MainMetric));
for v = 1:numel(VariantName)
    Parameter = Base;
    Field = fieldnames(VariantSetting{v});
    for f = 1:numel(Field), Parameter.(Field{f}) = VariantSetting{v}.(Field{f}); end
    Result = CrossValidation(Dataset, Parameter);
    for k = 1:numel(MainMetric)
        VariantMetric(v, k) = Result.MetricMean(strcmp(Result.MetricName, MainMetric{k}));
    end
end
fprintf('\n%-28s', 'Variant'); fprintf('%11s', MainMetric{:}); fprintf('\n');
for v = 1:numel(VariantName)
    fprintf('%-28s', VariantName{v}); fprintf('%11.4f', VariantMetric(v, :)); fprintf('\n');
end

ModalityName = Dataset.ModalityName;
Combination = {};
for s = 1:numel(ModalityName)
    Subset = nchoosek(1:numel(ModalityName), s);
    for i = 1:size(Subset, 1), Combination{end + 1} = ModalityName(Subset(i, :)); end
end
CombinationMetric = NaN(numel(Combination), numel(MainMetric));
for v = 1:numel(Combination)
    Parameter = Base;
    Parameter.Modality = Combination{v};
    Result = CrossValidation(Dataset, Parameter);
    for k = 1:numel(MainMetric)
        CombinationMetric(v, k) = Result.MetricMean(strcmp(Result.MetricName, MainMetric{k}));
    end
end
fprintf('\n%-28s', 'Modalities'); fprintf('%11s', MainMetric{:}); fprintf('\n');
for v = 1:numel(Combination)
    fprintf('%-28s', strjoin(Combination{v}, '+')); fprintf('%11.4f', CombinationMetric(v, :)); fprintf('\n');
end

Table = {'Ablation_Sensitivity.csv', 'InitSteady', arrayfun(@(x) sprintf('%g', x), InitSteady, 'UniformOutput', false), ...
         arrayfun(@(x) sprintf('AUROC_RegCoeff_%g', x), RegCoeff, 'UniformOutput', false), SensMean; ...
         'Ablation_Variant.csv', 'Variant', VariantName, MainMetric, VariantMetric; ...
         'Ablation_Modality.csv', 'Modalities', cellfun(@(c) strjoin(c, '+'), Combination, 'UniformOutput', false), ...
         MainMetric, CombinationMetric};
for t = 1:size(Table, 1)
    Fid = fopen(fullfile(OutFolder, Table{t, 1}), 'w');
    fprintf(Fid, '%s,%s\n', Table{t, 2}, strjoin(Table{t, 4}, ','));
    for i = 1:numel(Table{t, 3})
        fprintf(Fid, '%s%s\n', Table{t, 3}{i}, sprintf(',%.6f', Table{t, 5}(i, :)));
    end
    fclose(Fid);
end
save(fullfile(OutFolder, 'EMGPN_Ablation.mat'), 'InitSteady', 'RegCoeff', 'SensMean', 'SensStd', ...
    'VariantName', 'VariantMetric', 'Combination', 'CombinationMetric', 'MainMetric', '-v7');
fprintf('Results saved in %s\n', OutFolder);

Result = struct('Sensitivity',SensMean,'SensitivityStd',SensStd,'Variant',VariantMetric,'Modality',CombinationMetric);
end
