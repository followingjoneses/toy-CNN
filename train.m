load dataset X_train y_train;

m = 7780;
filter_num = 200;
image_dim = 28;
filter_dim = 8;
pool_dim = 3;
output_dim = 8;
p_size = (image_dim - filter_dim + 1)/pool_dim;
hidden_size = p_size^2*filter_num;

p = randperm(m);

X_train = X_train(:, :, p);
y_train = y_train(p, :);

Wc = 0.1*randn(filter_dim, filter_dim, filter_num);
bc = zeros(filter_num, 1);
r  = sqrt(6) / sqrt(output_dim+hidden_size);
Wd = rand(output_dim, hidden_size) * 2 * r - r;
bd = zeros(output_dim, 1);

epochs = 5;
minibatch = 100;
learning_rate = 0.08;
momentum = 0.5;
momentum_ = 0.95;
time = 40;

params = [Wc(:); Wd(:); bc(:); bd(:)];

velocity = zeros(size(params));

for epoch = 1:epochs
    rp = randperm(m);
    for batch = 1:minibatch:(m-minibatch+1)
        iteration = (batch-1)/minibatch+1;
        if iteration == time
            momentum = momentum_;
        end
        X_mini = X_train(:, :, rp(batch:batch+minibatch-1));
        y_mini = y_train(rp(batch:batch+minibatch-1), :);
        
        [L, grad] = costFunction(params, X_mini, y_mini, filter_dim, filter_num, pool_dim);
        
        velocity = velocity * momentum + learning_rate * grad;
        params = params - velocity;
        
        fprintf('Epoch %d: Cost on iteration %d is %f\n',epoch, iteration, L);
    end
    learning_rate = learning_rate/2;
end

fprintf('Training finished.\n');

[Wc, Wd, bc, bd] = unroll(params, filter_dim, filter_num, hidden_size, output_dim);

save params Wc Wd bc bd;