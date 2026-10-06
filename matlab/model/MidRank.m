function Rank = MidRank(Value)

if ~isnumeric(Value) || ~isreal(Value) || any(~isfinite(Value(:)))
    error('EMGPN:InvalidInput', 'MidRank requires finite real numeric values.');
end
Value = Value(:)';
n = numel(Value);
if n == 0, Rank = zeros(1, 0); return, end
[Sorted, Order] = sort(Value);
IsStart = [true, Sorted(2:end) ~= Sorted(1:end - 1)];
Start = find(IsStart);
Stop = [Start(2:end) - 1, n];
Group = cumsum(IsStart);
Rank = zeros(1, n);
Rank(Order) = (Start(Group) + Stop(Group)) / 2;
end
