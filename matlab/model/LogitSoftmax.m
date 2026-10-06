function Out = LogitSoftmax(S)

Shift = S - max(S, [], 1);
Expo = exp(Shift);
Denom = sum(Expo, 1);
Out.S = S;
Out.P = Expo ./ Denom;
Out.LogP = Shift - log(Denom);
end
