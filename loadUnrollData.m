% training dataset
load dataset X_train y_train;

m = 7780;
image_dim = 28;
p = randperm(m);

X = X_train;

X_train = zeros(m, image_dim*image_dim);

for i = 1:m
    temp = X(:, :, i)';
    temp = reshape(temp, 1, 784);
    X_train(i, :) = temp;
end

X_train = X_train(p, :);
y_train = y_train(p, :);

save unroll_dataset X_train y_train;

% validation dataset
load validation X_val y_val;

m = 3200;
p = randperm(m);
X = X_val;

X_val = zeros(m, image_dim*image_dim);

for i = 1:m
    temp = X(:, :, i)';
    temp = reshape(temp, 1, 784);
    X_val(i, :) = temp();
end
X_val = X_val(p, :);
y_val = y_val(p, :);

save unroll_validation X_val y_val;