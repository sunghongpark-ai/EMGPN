function Perf = PerformMeasure(P, y, varargin)

Opt = struct('Decision', 'argmax', 'Threshold', []);
if mod(numel(varargin), 2) ~= 0
    error('EMGPN:InvalidInput', 'Options must be name-value pairs.');
end
for i = 1:2:numel(varargin)
    if ~isfield(Opt, varargin{i}), error('EMGPN:InvalidInput', 'Unknown option %s.', varargin{i}); end
    Opt.(varargin{i}) = varargin{i + 1};
end
[NumClass, NumSubj] = size(P);
if ~isnumeric(P) || ~isreal(P) || ~ismatrix(P) || NumClass < 2 || NumSubj < 1
    error('EMGPN:InvalidInput', 'P must be a real c-by-n matrix with c >= 2 and n >= 1.');
end
if ~isnumeric(y) || ~isreal(y) || ~isvector(y)
    error('EMGPN:InvalidInput', 'y must be a real numeric label vector.');
end
y = double(y(:)');
if numel(y) ~= NumSubj || any(~isfinite(y) | y < 1 | y > NumClass | y ~= round(y))
    error('EMGPN:InvalidInput', 'y must hold %d labels in 1..%d.', NumSubj, NumClass);
end
if any(~isfinite(P(:)) | P(:) < 0 | P(:) > 1)
    error('EMGPN:InvalidInput', 'P must contain finite scores in [0,1].');
end
Y = zeros(NumClass, NumSubj);
Y(sub2ind(size(Y), y, 1:NumSubj)) = 1;
Present = sum(Y, 2)' > 0;
if isempty(Opt.Threshold), Opt.Threshold = 1 / NumClass; end
if ~(isnumeric(Opt.Threshold) && isreal(Opt.Threshold) && isscalar(Opt.Threshold) ...
        && isfinite(Opt.Threshold) && Opt.Threshold >= 0 && Opt.Threshold <= 1)
    error('EMGPN:InvalidInput', 'Threshold must be a finite scalar in [0,1].');
end

Perf.ClassAUROC = NaN(1, NumClass);
Perf.ClassAUPRC = NaN(1, NumClass);
for k = 1:NumClass
    Perf.ClassAUROC(k) = BinaryAuroc(P(k, :), Y(k, :));
    Perf.ClassAUPRC(k) = AveragePrecision(P(k, :), Y(k, :));
end
Valid = ~isnan(Perf.ClassAUROC);
Perf.AUROC = mean(Perf.ClassAUROC(Valid));
Perf.AUPRC = mean(Perf.ClassAUPRC(Valid));
Perf.MicroAUROC = BinaryAuroc(P(:)', Y(:)');
Perf.MicroAUPRC = AveragePrecision(P(:)', Y(:)');

[~, Yhat] = max(P, [], 1);
Perf.Accuracy = mean(Yhat == y);
Perf.Confusion = accumarray([y(:), Yhat(:)], 1, [NumClass, NumClass]);
switch lower(Opt.Decision)
    case 'argmax'
        TP = diag(Perf.Confusion)';
        FP = sum(Perf.Confusion, 1) - TP;
        FN = sum(Perf.Confusion, 2)' - TP;
    case 'ovr'
        D = P >= Opt.Threshold;
        TP = sum(D & Y == 1, 2)';
        FP = sum(D & Y == 0, 2)';
        FN = sum(~D & Y == 1, 2)';
    otherwise
        error('EMGPN:InvalidInput', 'Decision must be ''argmax'' or ''ovr''.');
end
Perf.ClassPrecision = SafeDivide(TP, TP + FP);
Perf.ClassRecall = SafeDivide(TP, TP + FN);
Perf.ClassF1 = SafeDivide(2 * TP, 2 * TP + FP + FN);
Label = Present | (TP + FP) > 0;
Perf.Precision = mean(Perf.ClassPrecision(Label));
Perf.Recall = mean(Perf.ClassRecall(Label));
Perf.F1 = mean(Perf.ClassF1(Label));
Recall = diag(Perf.Confusion)' ./ max(sum(Perf.Confusion, 2)', 1);
Perf.BalancedAccuracy = mean(Recall(Present));
Perf.LogLoss = -mean(log(max(P(sub2ind(size(P), y, 1:NumSubj)), realmin)));
end

function A = BinaryAuroc(Score, Label)

Label = Label > 0;
NumPos = nnz(Label);
NumNeg = numel(Label) - NumPos;
if NumPos == 0 || NumNeg == 0
    A = NaN;
    return
end
Rank = MidRank(Score);
A = (sum(Rank(Label)) - NumPos * (NumPos + 1) / 2) / (NumPos * NumNeg);
end

function AP = AveragePrecision(Score, Label)

Label = Label(:)' > 0;
NumPos = nnz(Label);
if NumPos == 0
    AP = NaN;
    return
end
[Sorted, Order] = sort(Score(:)', 'descend');
Hit = Label(Order);
TP = cumsum(Hit);
FP = cumsum(~Hit);
Last = [Sorted(1:end - 1) ~= Sorted(2:end), true];
TP = TP(Last);
FP = FP(Last);
Precision = TP ./ (TP + FP);
Recall = TP / NumPos;
AP = sum(diff([0, Recall]) .* Precision);
end

function Q = SafeDivide(A, B)
Q = zeros(size(A));
Q(B > 0) = A(B > 0) ./ B(B > 0);
end
