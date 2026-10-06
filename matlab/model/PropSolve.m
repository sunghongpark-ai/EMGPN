function Y = PropSolve(model, m, B)

R = model.PropChol{m};
if ~isnumeric(B) || ~isreal(B) || ~ismatrix(B) || size(B, 1) ~= model.NumRoi || any(~isfinite(B(:)))
    error('EMGPN:InvalidInput', 'Propagation RHS must be a finite real NumRoi-by-k matrix.');
end
Y = R \ (R' \ B);
if any(~isfinite(Y(:)))
    error('EMGPN:NonFiniteSolve', 'Propagation solve produced non-finite values.');
end
end
