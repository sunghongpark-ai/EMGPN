function Dataset = DataRead(File, varargin)

Opt = struct('IdColumn', 'SubjectID', 'LabelColumn', 'Diagnosis', 'ClassName', {{}}, ...
    'Modality', {{}}, 'Delimiter', ',');
if mod(numel(varargin), 2) ~= 0
    error('EMGPN:InvalidInput', 'Options must be name-value pairs.');
end
for i = 1:2:numel(varargin)
    if ~isfield(Opt, varargin{i}), error('EMGPN:InvalidInput', 'Unknown option %s.', varargin{i}); end
    Opt.(varargin{i}) = varargin{i + 1};
end
Opt.IdColumn = char(Opt.IdColumn);
Opt.LabelColumn = char(Opt.LabelColumn);
Opt.Delimiter = char(Opt.Delimiter);
if numel(Opt.Delimiter) ~= 1 || any(Opt.Delimiter == ['"', char(10), char(13)])
    error('EMGPN:InvalidInput', 'Delimiter must be one character other than a quote or line break.');
end
if exist(File, 'file') ~= 2
    error('EMGPN:FileError', 'File not found: %s', File);
end

Text = fileread(File);
if ~isempty(Text) && double(Text(1)) == 65279
    Text = Text(2:end);
elseif numel(Text) >= 3 && isequal(double(Text(1:3)), [239 187 191])
    Text = Text(4:end);
end
Line = CsvRecords(Text);
Line = Line(~cellfun(@(s) all(isspace(s)), Line));
if numel(Line) < 2
    error('EMGPN:InvalidInput', '%s needs a header and at least one participant row.', File);
end
Header = strtrim(SplitLine(Line{1}, Opt.Delimiter));
NumCol = numel(Header);
if numel(unique(Header)) ~= NumCol || any(cellfun(@isempty, Header))
    error('EMGPN:InvalidInput', 'Column names must be unique.');
end
NumSubj = numel(Line) - 1;
Field = cell(NumSubj, NumCol);
for j = 1:NumSubj
    Token = SplitLine(Line{j + 1}, Opt.Delimiter);
    if numel(Token) ~= NumCol
        error('EMGPN:InvalidInput', 'Row %d has %d fields; the header has %d.', j + 1, numel(Token), NumCol);
    end
    Field(j, :) = Token;
end

IdCol = FindColumn(Header, Opt.IdColumn, 'ID', false);
LabelCol = FindColumn(Header, Opt.LabelColumn, 'Label', true);
Free = setdiff(1:NumCol, [IdCol, LabelCol], 'stable');
[Prefix, Suffix] = cellfun(@SplitName, Header, 'UniformOutput', false);
if isempty(Opt.Modality)
    ModalityName = InferModality(Prefix(Free), Suffix(Free));
else
    ModalityName = cellstr(Opt.Modality);
    ModalityName = ModalityName(:)';
end
if numel(unique(ModalityName)) ~= numel(ModalityName) || any(cellfun(@(s) isempty(strtrim(s)), ModalityName))
    error('EMGPN:InvalidInput', 'Modality names must be nonempty and unique.');
end
M = numel(ModalityName);
ModalCol = cell(1, M);
for m = 1:M
    ModalCol{m} = Free(strcmp(Prefix(Free), ModalityName{m}));
    if isempty(ModalCol{m})
        error('EMGPN:InvalidInput', 'No columns for modality %s.', ModalityName{m});
    end
end
RoiName = Suffix(ModalCol{1})';
Xdata = cell(1, M);
for m = 1:M
    Name = Suffix(ModalCol{m});
    [Found, Loc] = ismember(RoiName, Name);
    if numel(Name) ~= numel(RoiName) || ~all(Found) || numel(unique(Name)) ~= numel(Name)
        error('EMGPN:InvalidInput', 'Modality %s does not have the ROIs of %s.', ModalityName{m}, ModalityName{1});
    end
    Col = ModalCol{m}(Loc);
    Xdata{m} = ToNumber(Field(:, Col), Header(Col))';
end

if isempty(IdCol)
    SubjectID = arrayfun(@(j) sprintf('P%04d', j), (1:NumSubj)', 'UniformOutput', false);
else
    SubjectID = strtrim(Field(:, IdCol));
end
if numel(unique(SubjectID)) ~= NumSubj || any(cellfun(@isempty, SubjectID))
    error('EMGPN:InvalidInput', 'Participant IDs must be nonempty and unique.');
end
if isempty(LabelCol)
    Ydata = [];
    ClassName = cellstr(Opt.ClassName);
else
    [Ydata, ClassName] = ParseLabel(strtrim(Field(:, LabelCol)), Opt.ClassName);
end
Used = [IdCol, LabelCol, ModalCol{:}];
Rest = setdiff(1:NumCol, Used, 'stable');
Covariate = struct('Name', {Header(Rest)}, 'Value', {cell(1, numel(Rest))});
for i = 1:numel(Rest)
    try
        Covariate.Value{i} = ToNumber(Field(:, Rest(i)), Header(Rest(i)));
    catch
        Covariate.Value{i} = strtrim(Field(:, Rest(i)));
    end
end
[~, Base, Ext] = fileparts(File);
Dataset = struct('Xdata', {Xdata}, 'Ydata', Ydata, 'ModalityName', {ModalityName}, ...
    'ClassName', {ClassName(:)'}, 'RoiName', {RoiName}, 'SubjectID', {SubjectID}, ...
    'Covariate', Covariate, 'Source', File, 'Description', ['Read from ', Base, Ext]);
end

function Token = SplitLine(Line, Delim)

if ~any(Line == '"')
    Token = regexp(Line, regexptranslate('escape', Delim), 'split');
    return
end
Token = {};
Buffer = '';
InQuote = false;
Closed = false;
i = 1;
while i <= numel(Line)
    ch = Line(i);
    if InQuote
        if ch == '"'
            if i < numel(Line) && Line(i + 1) == '"'
                Buffer(end + 1) = '"';
                i = i + 1;
            else
                InQuote = false;
                Closed = true;
            end
        else
            Buffer(end + 1) = ch;
        end
    elseif Closed && ch ~= Delim
        if ~isspace(ch)
            error('EMGPN:InvalidInput', 'Malformed CSV: characters after a closing quote.');
        end
    elseif ch == '"'
        if ~isempty(Buffer)
            error('EMGPN:InvalidInput', 'Malformed CSV: quote inside an unquoted field.');
        end
        InQuote = true;
    elseif ch == Delim
        Token{end + 1} = Buffer;
        Buffer = '';
        Closed = false;
    else
        Buffer(end + 1) = ch;
    end
    i = i + 1;
end
if InQuote
    error('EMGPN:InvalidInput', 'Malformed CSV: unterminated quoted field.');
end
Token{end + 1} = Buffer;
end

function Record = CsvRecords(Text)

if ~any(Text == '"')
    Record = regexp(Text, '\r\n|\n|\r', 'split');
    return
end
Record = {};
InQuote = false;
Start = 1;
i = 1;
while i <= numel(Text)
    ch = Text(i);
    if ch == '"'
        if InQuote && i < numel(Text) && Text(i + 1) == '"'
            i = i + 1;
        else
            InQuote = ~InQuote;
        end
    elseif ~InQuote && (ch == char(10) || ch == char(13))
        Record{end + 1} = Text(Start:i - 1);
        if ch == char(13) && i < numel(Text) && Text(i + 1) == char(10), i = i + 1; end
        Start = i + 1;
    end
    i = i + 1;
end
if InQuote
    error('EMGPN:InvalidInput', 'Malformed CSV: unterminated quoted field.');
end
Record{end + 1} = Text(Start:end);
end

function Col = FindColumn(Header, Name, Role, Required)
Col = zeros(1, 0);
if isempty(Name), return, end
Col = find(strcmp(Header, Name));
if isempty(Col) && Required
    error('EMGPN:InvalidInput', '%s column ''%s'' not found.', Role, Name);
end
end

function [Prefix, Suffix] = SplitName(Name)
Pos = find(Name == '_', 1);
if isempty(Pos) || Pos == 1 || Pos == numel(Name)
    Prefix = '';
    Suffix = '';
else
    Prefix = Name(1:Pos - 1);
    Suffix = Name(Pos + 1:end);
end
end

function ModalityName = InferModality(Prefix, Suffix)
Valid = ~cellfun(@isempty, Prefix);
Group = unique(Prefix(Valid), 'stable');
Candidate = {};
Key = {};
for i = 1:numel(Group)
    Member = Suffix(Valid & strcmp(Prefix, Group{i}));
    if numel(Member) >= 2
        Candidate{end + 1} = Group{i};
        Key{end + 1} = strjoin(sort(Member), char(31));
    end
end
if isempty(Candidate)
    error('EMGPN:InvalidInput', 'No imaging columns found; name them <Modality>_<ROI>, e.g. sMRI_ROI001.');
end
Best = 0;
BestScore = [-1, -1];
for i = 1:numel(Key)
    Score = [nnz(strcmp(Key, Key{i})), nnz(Key{i} == char(31)) + 1];
    if Score(1) > BestScore(1) || (Score(1) == BestScore(1) && Score(2) > BestScore(2))
        Best = i;
        BestScore = Score;
    end
end
Keep = strcmp(Key, Key{Best});
ModalityName = Candidate(Keep);
if any(~Keep)
    warning('EMGPN:IgnoredColumn', 'Column groups %s do not share the ROI set of %s and are kept as covariates.', ...
        strjoin(Candidate(~Keep), ', '), strjoin(ModalityName, ', '));
end
end

function Value = ToNumber(Field, Name)

Value = str2double(Field);
NotNumber = find(isnan(Value));
if ~isempty(NotNumber)
    Token = upper(strtrim(Field(NotNumber)));
    Missing = cellfun(@isempty, Token) | ismember(Token, {'NA', 'NAN', 'N/A', 'NULL'});
    if ~all(Missing)
        [Row, Col] = ind2sub(size(Field), NotNumber(find(~Missing, 1)));
        error('EMGPN:InvalidInput', 'Non-numeric value ''%s'' in row %d, column %s.', ...
            strtrim(Field{Row, Col}), Row + 1, Name{Col});
    end
end
if any(isinf(Value(:)))
    error('EMGPN:InvalidInput', 'Imaging values must be finite; leave missing values empty.');
end
end

function [Ydata, ClassName] = ParseLabel(Raw, ClassName)
if any(cellfun(@isempty, Raw) | ismember(upper(Raw), {'NA', 'NAN', 'N/A', 'NULL'}))
    error('EMGPN:InvalidInput', 'The label column has missing values.');
end
Code = str2double(Raw);
if any(isinf(Code))
    error('EMGPN:InvalidInput', 'The label column contains a non-finite numeric code.');
end
if ~isempty(ClassName)
    ClassName = cellstr(ClassName);
    ClassName = ClassName(:)';
    if numel(unique(ClassName)) ~= numel(ClassName) || any(cellfun(@(s) isempty(strtrim(s)), ClassName))
        error('EMGPN:InvalidInput', 'ClassName entries must be nonempty and unique.');
    end
    [Found, Ydata] = ismember(Raw, ClassName);
    if ~all(Found)
        if all(isfinite(Code)) && all(Code == round(Code)) && min(Code) >= 1 && max(Code) <= numel(ClassName)
            Ydata = Code;
        else
            Unknown = unique(Raw(~Found));
            error('EMGPN:InvalidInput', 'Labels not found in ClassName: %s', strjoin(Unknown(1:min(end, 10))', ', '));
        end
    end
elseif all(~isnan(Code))
    Level = unique(Code);
    [~, Ydata] = ismember(Code, Level);
    ClassName = arrayfun(@(v) sprintf('%g', v), Level(:)', 'UniformOutput', false);
else
    Level = unique(Raw);
    [~, Ydata] = ismember(Raw, Level);
    ClassName = Level(:)';
end
Ydata = double(Ydata(:)');
end
