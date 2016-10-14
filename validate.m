load params Wc Wd bc bd;
load validation X_val y_val;

m = size(X_val, 3);
y_val = y_val';

activations = convolve(X_val, Wc, bc);
activations_pooled = reshape(pool(pool_dim, activations), [], m);

probs = Wd * activations_pooled + repmat(bd, [1, m]);
probs = exp(probs);
probs = bsxfun(@rdivide, probs, sum(probs));

[maxv, labels] = max(y_val);
[maxp, preds] = max(probs);

correct = 0;

for i = 1:m
    if labels(i) == preds(i)
        correct = correct + 1;
    end
end

fprintf('Accuracy = %f\n', correct/m);