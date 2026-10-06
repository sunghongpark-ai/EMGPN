function [W, Sigma, Neighbor] = GraphKnn(X, NumNeighbor, Sigma)

[NumRoi, ~] = size(X);
if NumRoi < 2
    error('EMGPN:InvalidInput', 'A graph needs at least two nodes.');
end
if any(~isfinite(X(:)))
    error('EMGPN:InvalidInput', 'Graph construction requires finite (imputed) data.');
end
k = min(NumNeighbor, NumRoi - 1);

Xc = X - mean(X, 1);
SqNorm = sum(Xc.^2, 2);
D2 = SqNorm + SqNorm' - 2 * (Xc * Xc');
D2 = max((D2 + D2') / 2, 0);
D2(1:NumRoi + 1:end) = Inf;

[~, Order] = sort(D2, 2, 'ascend');
Neighbor = Order(:, 1:k);
Directed = false(NumRoi, NumRoi);
Directed(sub2ind([NumRoi, NumRoi], repmat((1:NumRoi)', k, 1), Neighbor(:))) = true;

if nargin < 3 || isempty(Sigma)
    Sigma = mean(sqrt(D2(Directed)));
    if ~(Sigma > 0), Sigma = 1; end
end
Edge = Directed | Directed';
W = zeros(NumRoi, NumRoi);
W(Edge) = exp(-D2(Edge) / Sigma^2);
end
