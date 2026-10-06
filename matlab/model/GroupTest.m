function [PValue, Stat] = GroupTest(Z, y)

[NumRow, NumSubj] = size(Z);
if ~isnumeric(Z) || ~isreal(Z) || ~ismatrix(Z) || any(~isfinite(Z(:))) ...
        || ~isnumeric(y) || ~isreal(y) || any(~isfinite(y(:)))
    error('EMGPN:InvalidInput', 'Z and y must contain finite real numeric values.');
end
y = y(:)';
if numel(y) ~= NumSubj
    error('EMGPN:InvalidInput', 'y must have %d elements.', NumSubj);
end
PValue = NaN(NumRow, 1);
Stat = NaN(NumRow, 1);
if NumSubj == 0, return, end
[~, ~, GroupIdx] = unique(y);
GroupIdx = GroupIdx(:);
NumGroup = max(GroupIdx);
Count = accumarray(GroupIdx, 1, [NumGroup, 1]);
if NumGroup < 2 || NumSubj < 3
    return
end
for q = 1:NumRow
    Rank = MidRank(Z(q, :));
    SumRank = accumarray(GroupIdx, Rank(:), [NumGroup, 1]);
    H = 12 / (NumSubj * (NumSubj + 1)) * sum(SumRank .^ 2 ./ Count) - 3 * (NumSubj + 1);
    Sorted = sort(Z(q, :));
    Start = find([true, Sorted(2:end) ~= Sorted(1:end - 1)]);
    Tie = diff([Start, NumSubj + 1]);
    Correction = 1 - sum(Tie .^ 3 - Tie) / (NumSubj ^ 3 - NumSubj);
    if Correction > 0
        H = max(H / Correction, 0);
        PValue(q) = gammainc(H / 2, (NumGroup - 1) / 2, 'upper');
    else
        H = 0;
        PValue(q) = 1;
    end
    Stat(q) = H;
end
end
