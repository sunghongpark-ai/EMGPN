function model = GraphConstruct(model)

M = model.NumModal;
r = model.NumRoi;
model.Wgraph = cell(1, M);
model.Lgraph = cell(1, M);
model.Agcn = cell(1, M);
model.Sigma = NaN(1, M);
for m = 1:M
    if ~isempty(model.Wdata) && ~isempty(model.Wdata{m})
        W = double(model.Wdata{m});
        if ~isequal(size(W), [r, r]) || ~isreal(W) || any(~isfinite(W(:))) || any(W(:) < 0)
            error('EMGPN:InvalidInput', 'Wdata{%d} must be a finite nonnegative %d-by-%d matrix.', m, r, r);
        end
        W = (W + W') / 2;
        W(1:r + 1:end) = 0;
    else
        KW = model.KernelWidth;
        if isempty(KW)
            Sigma = [];
        elseif isscalar(KW)
            Sigma = KW;
        elseif numel(KW) == model.NumModalTotal
            Sigma = KW(model.ModalityIndex(m));
        else
            Sigma = KW(m);
        end
        [W, Sigma] = GraphKnn(model.Xtrain{m}, model.NumNeighbor, Sigma);
        model.Sigma(m) = Sigma;
    end
    [L, Ahat] = GraphLaplacian(W);
    model.Wgraph{m} = W;
    model.Lgraph{m} = L;
    if strcmp(model.Propagation, 'gcn')
        model.Agcn{m} = Ahat;
    end
end
end
