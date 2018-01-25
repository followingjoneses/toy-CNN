load params Wc Wd bc bd;
load testset X_test;

m = size(X_test, 3);

activations = convolve(X_test, Wc, bc);
activations_pooled = reshape(pool(3, activations), [], m);

probs = Wd * activations_pooled + repmat(bd, [1, m]);
probs = exp(probs);
probs = bsxfun(@rdivide, probs, sum(probs));

[maxp, preds] = max(probs);

f = fopen('answer/answer.txt', 'w+');
for i = 1:m
    if preds(i) == 1
        fprintf(f,'A\n');
    elseif preds(i) == 2
        fprintf(f,'B\n');
    elseif preds(i) == 3
        fprintf(f,'C\n');
    elseif preds(i) == 4
        fprintf(f,'D\n');
    elseif preds(i) == 5
        fprintf(f,'E\n');
    elseif preds(i) == 6
        fprintf(f,'F\n');
    elseif preds(i) == 7
        fprintf(f,'G\n');
    elseif preds(i) == 8
        fprintf(f,'H\n');
    end
end