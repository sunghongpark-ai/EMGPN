function [L, Ahat] = GraphLaplacian(W)

NumRoi = size(W, 1);
if ~isnumeric(W) || ~isreal(W) || ~ismatrix(W) || size(W, 2) ~= NumRoi ...
        || any(~isfinite(nonzeros(W))) || any(nonzeros(W) < 0) || ~isequal(W, W')
    error('EMGPN:InvalidInput', 'W must be a finite, nonnegative, symmetric square matrix.');
end
Degree = sum(W, 2);
s = zeros(NumRoi, 1);
s(Degree > 0) = 1 ./ sqrt(Degree(Degree > 0));
if issparse(W)

    [Row, Col, Value] = find(W);
    N = sparse(Row, Col, Value .* (s(Row) .* s(Col)), NumRoi, NumRoi);
    I = speye(NumRoi);
else
    N = W .* (s * s');
    I = eye(NumRoi);
end
L = I - N;
if nargout > 1
    Wt = W + I;
    st = 1 ./ sqrt(sum(Wt, 2));
    if issparse(Wt)
        [Row, Col, Value] = find(Wt);
        Ahat = sparse(Row, Col, Value .* (st(Row) .* st(Col)), NumRoi, NumRoi);
    else
        Ahat = Wt .* (st * st');
    end
end
end
