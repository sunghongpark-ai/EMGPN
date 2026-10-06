function Dataset = DataSimulate(varargin)

Opt = struct('NumRoi', 263, 'ClassSize', [80 237 139 124 64], 'Seed', 2025, ...
    'MissingRate', [0 0.03 0.02], 'DropoutRate', 0.02, 'SignalScale', 1, 'NoiseScale', 1);
Opt = ParseOption(Opt, varargin);
if ~(isnumeric(Opt.NumRoi) && isreal(Opt.NumRoi) && isscalar(Opt.NumRoi) && isfinite(Opt.NumRoi) ...
        && Opt.NumRoi >= 12 && Opt.NumRoi == round(Opt.NumRoi))
    error('EMGPN:InvalidInput', 'NumRoi must be an integer >= 12.');
end
if ~isnumeric(Opt.ClassSize) || ~isreal(Opt.ClassSize) || ~isvector(Opt.ClassSize) || any(~isfinite(Opt.ClassSize))
    error('EMGPN:InvalidInput', 'ClassSize must hold 2 to 5 positive integers.');
end
if ~isnumeric(Opt.MissingRate) || ~isreal(Opt.MissingRate) || numel(Opt.MissingRate) ~= 3 ...
        || any(~isfinite(Opt.MissingRate)) || any(Opt.MissingRate < 0 | Opt.MissingRate > 1)
    error('EMGPN:InvalidInput', 'MissingRate must contain three finite rates in [0, 1].');
end
for Name = {'DropoutRate', 'SignalScale', 'NoiseScale'}
    Value = Opt.(Name{1});
    if ~(isnumeric(Value) && isreal(Value) && isscalar(Value) && isfinite(Value) && Value >= 0)
        error('EMGPN:InvalidInput', '%s must be a finite nonnegative scalar.', Name{1});
    end
end
if Opt.DropoutRate > 1 || Opt.DropoutRate + Opt.MissingRate(2) > 1
    error('EMGPN:InvalidInput', 'DropoutRate + fMRI MissingRate must not exceed 1.');
end

ClassName = {'SCD', 'MCI', 'AD', 'VCI', 'VD'};
ClassSizeRef = [80 237 139 124 64];
FemaleCount = [64 163 95 83 40];
AmyloidCount = [0 44 129 8 16];
WmhCount = [59 21 0; 234 2 1; 93 42 4; 14 101 9; 2 39 23];
AgeMedian = [72 71 75 76 77];
AgeIqr = [9 10 10 9 10];
CdrsbMedian = [1.0 1.5 4.0 1.5 4.5];
CdrsbIqr = [0.5 1.5 3.0 2.0 3.5];
IqrToSd = 1.349;

ModalityName = {'sMRI', 'fMRI', 'PET'};
GroupName = {'VIS', 'SMN', 'DAN', 'VAN', 'LIM', 'FPN', 'DMN', 'HIP', 'AMY', 'BG', 'THA', 'CBL'};
GroupSize = [30 36 26 24 12 26 46 4 4 12 16 27];
NumCortical = 7;
PatternAmyloid = [0.00 0.00 0.90 0.85 0.80 0.10 1.20 0.10 0.60 0.85 0.00 0.00]';
PatternAtrophy = [0.55 0.10 0.05 0.05 0.15 0.05 0.10 1.10 0.20 0.10 0.40 0.05]';
PatternVascular = [0.05 0.80 0.05 0.05 0.05 0.05 0.05 0.10 0.05 0.20 0.90 0.05]';
PatternFunction = [0.05 0.05 0.05 0.20 0.05 1.10 0.15 0.05 0.05 0.05 0.05 0.40]';
PatternPerfusion = [0.05 0.05 0.05 0.05 0.05 0.50 0.05 0.05 0.05 0.05 0.05 0.30]';
SubtypeEffect = {'sMRI', 'VIS', 'SCD', 0.30; 'sMRI', 'SMN', 'SCD', 0.30; ...
    'sMRI', 'THA', 'MCI', -0.35; 'fMRI', 'FPN', 'MCI', -0.35; ...
    'sMRI', 'HIP', 'AD', -0.25; 'PET', 'DMN', 'AD', 0.25; ...
    'sMRI', 'SMN', 'VCI', -0.35; 'fMRI', 'CBL', 'VCI', -0.30; ...
    'sMRI', 'HIP', 'VD', -0.35; 'fMRI', 'FPN', 'VD', -0.30};
KappaAmyloid = 3.0;
KappaAtrophy = 0.9;
KappaVascular = 0.6;
KappaFunction = 1.0;
KappaPerfusion = 0.6;
KappaSubtype = 1.0;
KappaAge = 0.35;
KappaSex = 0.3;
NoiseRoi = [1.0 1.1 0.9];
NoiseGlobal = 0.3;

ClassSize = double(Opt.ClassSize(:)');
c = numel(ClassSize);
if c < 2 || c > numel(ClassName) || any(ClassSize < 1) || any(ClassSize ~= round(ClassSize))
    error('EMGPN:InvalidInput', 'ClassSize must hold 2 to 5 positive integers.');
end
if numel(Opt.MissingRate) ~= numel(ModalityName)
    error('EMGPN:InvalidInput', 'MissingRate must have one value per modality.');
end
n = sum(ClassSize);
[RoiName, Group] = RoiLayout(Opt.NumRoi, GroupName, GroupSize);
r = numel(RoiName);
NumGroup = numel(GroupName);
Kappa = Opt.SignalScale;
Stream = RandomStream(Opt.Seed);

Label = repelem(1:c, ClassSize);
[~, Order] = sort(RandomUniform(Stream, 1, n));
y = Label(Order);
Female = zeros(1, n);
Amyloid = zeros(1, n);
Wmh = zeros(1, n);
for k = 1:c
    Member = find(y == k);
    nk = numel(Member);
    Perm = Member(RandomOrder(Stream, nk));
    Female(Perm(1:ClassCount(FemaleCount(k), ClassSizeRef(k), nk))) = 1;
    Perm = Member(RandomOrder(Stream, nk));
    Amyloid(Perm(1:ClassCount(AmyloidCount(k), ClassSizeRef(k), nk))) = 1;
    Perm = Member(RandomOrder(Stream, nk));
    Mild = ClassCount(WmhCount(k, 1), ClassSizeRef(k), nk);
    Moderate = min(ClassCount(WmhCount(k, 2), ClassSizeRef(k), nk), nk - Mild);
    Wmh(Perm(Mild + 1:Mild + Moderate)) = 1;
    Wmh(Perm(Mild + Moderate + 1:end)) = 2;
end
Z = NormalQuantile(RandomUniform(Stream, 1, n));
Age = floor(min(max(AgeMedian(y) + AgeIqr(y) / IqrToSd .* Z, 50), 95) + 0.5);
Z = NormalQuantile(RandomUniform(Stream, 1, n));
Severity = min(max(CdrsbMedian(y) + CdrsbIqr(y) / IqrToSd .* Z, 0), 18);
Z = NormalQuantile(RandomUniform(Stream, 1, n));
Burden = Amyloid .* max(1 + 0.25 * Z, 0.3);
SevC = (Severity - 1.5) / 2.5;
Drive = SevC .* (0.5 + 0.5 * Amyloid);
AgeC = (Age - 74) / 7;

Cortical = Group <= NumCortical;
Cerebellum = Group == find(strcmp(GroupName, 'CBL'));
Xdata = cell(1, numel(ModalityName));
Information = zeros(r, numel(ModalityName));
for m = 1:numel(ModalityName)
    Zbase = NormalQuantile(RandomUniform(Stream, r, 1));
    Zjit = NormalQuantile(RandomUniform(Stream, r, 1));
    Uload = RandomUniform(Stream, r, 1);
    Znet = NormalQuantile(RandomUniform(Stream, NumGroup, n));
    Zglob = NormalQuantile(RandomUniform(Stream, 1, n));
    Zroi = NormalQuantile(RandomUniform(Stream, r, n));
    Jitter = max(1 + 0.2 * Zjit, 0);
    Load = 0.5 + 0.3 * Uload;
    Table = SubtypeTable(SubtypeEffect, ModalityName{m}, GroupName, ClassName, c);
    Subtype = (Kappa * KappaSubtype) * Table(Group, y) .* Jitter;
    switch ModalityName{m}
        case 'sMRI'
            Base = min(max(0.55 + 0.07 * Zbase, 0.30), 0.80);
            Scale = 0.09 * Base;
            Wa = (Kappa * KappaAtrophy) * PatternAtrophy(Group) .* Jitter;
            Wv = (Kappa * KappaVascular) * PatternVascular(Group) .* Jitter;
            Effect = Subtype - Wa .* Drive - Wv .* Wmh - (Kappa * KappaAge) * AgeC ...
                - (Kappa * KappaSex) * Female;
        case 'fMRI'
            Base = min(max(0.30 + 0.025 * Zbase, 0.18), 0.42);
            Scale = 0.03;
            Wf = (Kappa * KappaFunction) * PatternFunction(Group) .* Jitter;
            Wp = (Kappa * KappaPerfusion) * PatternPerfusion(Group) .* Jitter;
            Effect = Subtype - Wf .* SevC - Wp .* Wmh;
        otherwise
            Mean = 1.35 * ones(r, 1);
            Mean(Cortical) = 1.15;
            Mean(Cerebellum) = 1.00;
            Spread = 0.08 * ones(r, 1);
            Spread(Cortical) = 0.05;
            Spread(Cerebellum) = 0.01;
            Base = Mean + Spread .* Zbase;
            Scale = 0.06 * ones(r, 1);
            Scale(Cerebellum) = 0.015;
            Wb = (Kappa * KappaAmyloid) * PatternAmyloid(Group) .* Jitter;
            Effect = Subtype + Wb .* Burden;
    end
    Tau = Opt.NoiseScale * NoiseRoi(m);
    Glob = Opt.NoiseScale * NoiseGlobal;
    Noise = Load .* Znet(Group, :) + Glob * Zglob + Tau * Zroi;
    Xdata{m} = Base + Scale .* (Effect + Noise);
    Information(:, m) = InformationShare(Effect, y, c, Load .^ 2 + Glob ^ 2 + Tau ^ 2);
end

for m = 1:numel(ModalityName)
    if Opt.MissingRate(m) > 0
        Order = RandomOrder(Stream, n);
        Xdata{m}(:, Order(1:floor(Opt.MissingRate(m) * n + 0.5))) = NaN;
    end
end
if Opt.DropoutRate > 0
    f = find(strcmp(ModalityName, 'fMRI'));
    Order = RandomOrder(Stream, n);
    Order = Order(~isnan(Xdata{f}(1, Order)));
    NumDropout = floor(Opt.DropoutRate * n + 0.5);
    if NumDropout > numel(Order)
        error('EMGPN:InvalidInput', 'Rounded dropout count exceeds participants with observed fMRI.');
    end
    Xdata{f}(Group == find(strcmp(GroupName, 'LIM')), Order(1:NumDropout)) = NaN;
end

[~, TrueModality] = max(Information, [], 2);
Sex = repmat({'M'}, n, 1);
Sex(Female > 0) = {'F'};
Dataset.Xdata = Xdata;
Dataset.Ydata = y;
Dataset.ModalityName = ModalityName;
Dataset.ClassName = ClassName(1:c);
Dataset.RoiName = RoiName;
Dataset.SubjectID = arrayfun(@(j) sprintf('S%04d', j), (1:n)', 'UniformOutput', false);
Dataset.Covariate = struct('Name', {{'Age', 'Sex'}}, 'Value', {{Age(:), Sex}});
Dataset.Truth = struct('Group', Group, 'GroupName', {GroupName}, 'Amyloid', Amyloid, ...
    'Burden', Burden, 'WmhGrade', Wmh, 'Severity', Severity, 'Age', Age, 'Female', Female, ...
    'Information', Information, 'TrueModality', TrueModality);
Dataset.Description = sprintf(['Synthetic EMGPN sample (DataSimulate, seed %d): %d ROIs, ', ...
    '%d participants, %d subtypes, %d modalities'], Opt.Seed, r, n, c, numel(ModalityName));
end

function Opt = ParseOption(Opt, Arg)
if mod(numel(Arg), 2) ~= 0
    error('EMGPN:InvalidInput', 'Options must be name-value pairs.');
end
for i = 1:2:numel(Arg)
    Name = Arg{i};
    if ~(ischar(Name) || (isstring(Name) && isscalar(Name))) || ~isfield(Opt, char(Name))
        error('EMGPN:InvalidInput', 'Unknown option %s.', char(string(Name)));
    end
    Opt.(char(Name)) = Arg{i + 1};
end
end

function Order = RandomOrder(Stream, NumItem)
[~, Order] = sort(RandomUniform(Stream, 1, NumItem));
end

function Count = ClassCount(Reference, Size, NumMember)
Count = floor(Reference / Size * NumMember + 0.5);
end

function [Name, Group] = RoiLayout(NumRoi, GroupName, GroupSize)
NumGroup = numel(GroupSize);
Count = GroupSize;
if NumRoi ~= sum(GroupSize)
    if ~(isscalar(NumRoi) && NumRoi >= NumGroup && NumRoi == round(NumRoi))
        error('EMGPN:InvalidInput', 'NumRoi must be an integer >= %d.', NumGroup);
    end
    Quota = (NumRoi - NumGroup) * GroupSize / sum(GroupSize);
    Count = 1 + floor(Quota);
    [~, Order] = sort(-(Quota - floor(Quota)));
    Extra = NumRoi - sum(Count);
    Count(Order(1:Extra)) = Count(Order(1:Extra)) + 1;
end
Name = cell(sum(Count), 1);
Group = zeros(sum(Count), 1);
q = 0;
for g = 1:NumGroup
    t = Count(g);
    if strcmp(GroupName{g}, 'CBL') && mod(t, 2) == 1
        Hemi = {'L', 'R', 'V'};
        Num = [(t - 1) / 2, (t - 1) / 2, 1];
    else
        Hemi = {'L', 'R'};
        Num = [ceil(t / 2), floor(t / 2)];
    end
    for h = 1:numel(Hemi)
        for i = 1:Num(h)
            q = q + 1;
            Name{q} = sprintf('%s_%s%02d', GroupName{g}, Hemi{h}, i);
            Group(q) = g;
        end
    end
end
end

function Table = SubtypeTable(Effect, Modality, GroupName, ClassName, NumClass)
Table = zeros(numel(GroupName), NumClass);
for i = 1:size(Effect, 1)
    k = find(strcmp(ClassName, Effect{i, 3}));
    if strcmp(Effect{i, 1}, Modality) && k <= NumClass
        Table(strcmp(GroupName, Effect{i, 2}), k) = Effect{i, 4};
    end
end
end

function Share = InformationShare(Effect, y, NumClass, NoiseVar)

n = size(Effect, 2);
Grand = mean(Effect, 2);
Between = zeros(size(Effect, 1), 1);
Within = zeros(size(Effect, 1), 1);
for k = 1:NumClass
    Ek = Effect(:, y == k);
    if isempty(Ek), continue, end
    Mu = mean(Ek, 2);
    Between = Between + size(Ek, 2) * (Mu - Grand) .^ 2;
    Within = Within + sum((Ek - Mu) .^ 2, 2);
end
Share = (Between / n) ./ (Between / n + Within / n + NoiseVar);
end
