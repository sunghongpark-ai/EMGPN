function Key = KeyRegion(FeatureImportance, PermutationImportance, ProbChange, GroupLogP)

FI = FeatureImportance(:);
PI = PermutationImportance(:);
if nargin < 4 || isempty(GroupLogP), GroupLogP = NaN(size(FI)); end
Key.FeatureImportanceNorm = MinMaxScale(FI);
Key.PermutationImportanceNorm = MinMaxScale(PI);
Key.CombinedImportance = (Key.FeatureImportanceNorm + Key.PermutationImportanceNorm) / 2;
Key.IsKey = FI > mean(FI) & PI > mean(PI);
KeyIdx = find(Key.IsKey);
[~, Order] = sort(Key.CombinedImportance(KeyIdx), 'descend');
Key.KeyRoi = KeyIdx(Order);
NumClass = size(ProbChange, 2);
Key.SubtypeKeyRoi = cell(1, NumClass);
for k = 1:NumClass
    Key.SubtypeKeyRoi{k} = Key.KeyRoi(ProbChange(Key.KeyRoi, k) > 0);
end
Key.ProbChangeKey = MeanRows(ProbChange, Key.IsKey);
Key.ProbChangeRest = MeanRows(ProbChange, ~Key.IsKey);
Key.GroupLogPKey = MeanRows(GroupLogP(:), Key.IsKey);
Key.GroupLogPRest = MeanRows(GroupLogP(:), ~Key.IsKey);
end

function x = MinMaxScale(x)
Range = max(x) - min(x);
if Range > 0
    x = (x - min(x)) / Range;
else
    x = zeros(size(x));
end
end

function Mu = MeanRows(A, Rows)
if any(Rows)
    Mu = mean(A(Rows, :), 1);
else
    Mu = NaN(1, size(A, 2));
end
end
