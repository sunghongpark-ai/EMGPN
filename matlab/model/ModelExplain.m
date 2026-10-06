function Explain = ModelExplain(model, X, y, varargin)

Opt = struct('GroupTest', true);
if mod(numel(varargin), 2) ~= 0
    error('EMGPN:InvalidInput', 'Options must be name-value pairs.');
end
for i = 1:2:numel(varargin)
    if ~isfield(Opt, varargin{i}), error('EMGPN:InvalidInput', 'Unknown option %s.', varargin{i}); end
    Opt.(varargin{i}) = varargin{i + 1};
end
if strcmp(model.Integration, 'concat')
    error('EMGPN:Unsupported', 'ModelExplain requires ROI-level integrated features (Integration ''softmax'' or ''mean'').');
end
model = ParamReshape(model);
Out = ModelForward(model, X);
Z = Out.Z;
S = Out.S;
P = Out.P;
y = double(y(:)');
[NumRoi, NumSubj] = size(Z);
NumClass = model.NumClass;
Theta = model.Theta;
Zbar = model.ZMean;
if NumSubj == 0 || numel(y) ~= NumSubj || any(~isfinite(y) | y < 1 | y > NumClass | y ~= round(y))
    error('EMGPN:InvalidInput', 'y must hold %d finite labels in 1..%d.', NumSubj, NumClass);
end

Part.ModalityName = model.ModalityName;
Part.ClassName = model.ClassName;
Part.RoiName = model.RoiName;
Part.ModalityImportance = model.LambdaProb;
if strcmp(model.Propagation, 'gpn'), Part.Steadiness = model.Phi; else, Part.Steadiness = []; end
Part.RiskEffect = Theta;

Part.FeatureImportance = mean(abs(Theta), 2) .* mean(abs(Z - Zbar), 2);

Stream = RandomStream(mod(model.RandSeed + 7919, 2^32));
Base = PermScore(S, y, model.PermMetric);
PI = zeros(NumRoi, 1);
for q = 1:NumRoi
    [~, Perm] = sort(RandomUniform(Stream, NumSubj, model.NumPermute), 1);
    Theta_q = Theta(q, :)';
    for b = 1:model.NumPermute
        PI(q) = PI(q) + (PermScore(S + Theta_q * (Z(q, Perm(:, b)) - Z(q, :)), y, model.PermMetric) - Base);
    end
end
if strcmp(model.PermMetric, 'auroc'), PI = -PI; end
Part.PermutationImportance = PI / model.NumPermute;

if strcmp(model.ProbBaseline, 'mean'), Baseline = Zbar; else, Baseline = zeros(NumRoi, 1); end
Part.ProbChange = NaN(NumRoi, NumClass);
Part.ProbChangeAbs = NaN(NumRoi, NumClass);
for q = 1:NumRoi
    Removed = LogitSoftmax(S - Theta(q, :)' * (Z(q, :) - Baseline(q)));
    for k = 1:NumClass
        if strcmp(model.ProbSubset, 'class'), Sub = (y == k); else, Sub = true(1, NumSubj); end
        if any(Sub)
            Part.ProbChange(q, k) = 100 * mean(expm1(Out.LogP(k, Sub) - Removed.LogP(k, Sub)));
            Part.ProbChangeAbs(q, k) = 100 * mean(P(k, Sub) - Removed.P(k, Sub));
        end
    end
end

if Opt.GroupTest
    Part.GroupPValue = GroupTest(Z, y);
else
    Part.GroupPValue = NaN(NumRoi, 1);
end
Explain = ExplainSummary(Part);
end

function Score = PermScore(S, y, Metric)
switch Metric
    case 'loss'
        Shift = S - max(S, [], 1);
        LogP = Shift - log(sum(exp(Shift), 1));
        Score = -mean(LogP(sub2ind(size(LogP), y, 1:numel(y))));
    otherwise
        Out = LogitSoftmax(S);
        Perf = PerformMeasure(Out.P, y);
        Score = Perf.AUROC;
end
end
