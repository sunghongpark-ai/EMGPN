function Stream = RandomStream(Seed)

if ~(isnumeric(Seed) && isscalar(Seed) && isfinite(Seed) && Seed >= 0 && Seed < 2^32 && Seed == round(Seed))
    error('EMGPN:InvalidInput', 'Seed must be an integer in [0, 2^32).');
end
Seed = double(Seed);
if Seed == 0
    Seed = 5489;
end
if exist('OCTAVE_VERSION', 'builtin') == 0
    Stream = RandStream('mt19937ar', 'Seed', Seed);
else
    rand('twister', TwisterState(Seed));
    Stream = struct('Seed', Seed);
end
end

function State = TwisterState(Seed)

State = zeros(625, 1);
State(1) = Seed;
for i = 2:624
    x = bitxor(State(i - 1), floor(State(i - 1) / 2^30));
    Hi = floor(x / 65536);
    Lo = x - Hi * 65536;
    State(i) = mod(1812433253 * Lo + mod(1812433253 * Hi, 65536) * 65536 + (i - 1), 2^32);
end
State(625) = 1;
State = uint32(State);
end
