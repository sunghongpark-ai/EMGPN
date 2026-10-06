function File = ResultExport(Result, Folder)

if nargin < 2 || isempty(Folder), Folder = 'Result'; end
if exist(Folder, 'dir') ~= 7, mkdir(Folder); end
File = cell(1, 0);

Name = fullfile(Folder, 'Performance_Repetition.csv');
Fid = OpenFile(Name);
fprintf(Fid, 'Repetition,%s\n', strjoin(Result.MetricName, ','));
for i = 1:size(Result.Metric, 1)
    fprintf(Fid, '%d%s\n', i, sprintf(',%.6f', Result.Metric(i, :)));
end
fclose(Fid);
File{end + 1} = Name;

Name = fullfile(Folder, 'Performance_Summary.csv');
Fid = OpenFile(Name);
fprintf(Fid, 'Metric,Mean,SD\n');
for k = 1:numel(Result.MetricName)
    fprintf(Fid, '%s,%.6f,%.6f\n', Result.MetricName{k}, Result.MetricMean(k), Result.MetricStd(k));
end
fclose(Fid);
File{end + 1} = Name;

Name = fullfile(Folder, 'Risk_OutOfFold.csv');
Fid = OpenFile(Name);
fprintf(Fid, 'SubjectID,Diagnosis%s,Predicted\n', sprintf(',Risk_%s', Result.ClassName{:}));
[~, Pred] = max(Result.RiskMean, [], 1);
for j = 1:numel(Result.Ydata)
    fprintf(Fid, '%s,%s%s,%s\n', Quote(Result.SubjectID{j}), Result.ClassName{Result.Ydata(j)}, ...
        sprintf(',%.6f', Result.RiskMean(:, j)), Result.ClassName{Pred(j)});
end
fclose(Fid);
File{end + 1} = Name;

Ex = Result.Explain;
if ~isempty(Ex)
    Name = fullfile(Folder, 'Explain_ROI.csv');
    Fid = OpenFile(Name);
    Header = [{'Index', 'ROI'}, strcat('Lambda_', Ex.ModalityName), {'DominantModality'}];
    if ~isempty(Ex.Steadiness), Header = [Header, strcat('Phi_', Ex.ModalityName)]; end
    Header = [Header, strcat('Theta_', Ex.ClassName), {'FeatureImportance', ...
        'PermutationImportance', 'CombinedImportance', 'IsKey'}, ...
        strcat('ProbChange_', Ex.ClassName), {'GroupPValue', 'GroupLogP'}];
    fprintf(Fid, '%s\n', strjoin(Header, ','));
    for q = 1:numel(Ex.RoiName)
        if isnan(Ex.DominantModality(q))
            Dominant = 'NA';
        else
            Dominant = Ex.ModalityName{Ex.DominantModality(q)};
        end
        Line = sprintf('%d,%s%s,%s', q, Quote(Ex.RoiName{q}), sprintf(',%.6f', Ex.ModalityImportance(q, :)), Dominant);
        if ~isempty(Ex.Steadiness)
            Line = [Line, sprintf(',%.6f', Ex.Steadiness(q, :))];
        end
        Line = [Line, sprintf(',%.6f', Ex.RiskEffect(q, :)), ...
            sprintf(',%.6g,%.6g,%.6f,%d', Ex.FeatureImportance(q), Ex.PermutationImportance(q), ...
            Ex.CombinedImportance(q), Ex.IsKey(q)), sprintf(',%.4f', Ex.ProbChange(q, :)), ...
            sprintf(',%.6g,%.4f', Ex.GroupPValue(q), Ex.GroupLogP(q))];
        fprintf(Fid, '%s\n', Line);
    end
    fclose(Fid);
    File{end + 1} = Name;
end
end

function Fid = OpenFile(Name)
[Fid, Msg] = fopen(Name, 'w');
if Fid < 0, error('EMGPN:FileError', 'Cannot open %s for writing: %s', Name, Msg); end
end

function s = Quote(s)
if any(s == ',') || any(s == '"')
    s = ['"', strrep(s, '"', '""'), '"'];
end
end
