function [L, grad] = costFunction(params, X, y, filter_dim, filter_num, pool_dim)
    image_dim = size(X, 1);
    m = size(X, 3);
    output_dim = size(y, 2);
    
    cov_dim = image_dim - filter_dim + 1;
    p_size = cov_dim/pool_dim;
    hidden_size = p_size^2 * filter_num;
    
    [Wc, Wd, bc, bd] = unroll(params, filter_dim, filter_num, hidden_size, output_dim);
    
    Wc_grad = zeros(size(Wc));
    bc_grad = zeros(size(bc));
    Wd_grad = zeros(size(Wd));
    bd_grad = zeros(size(bd));
    
    activations = convolve(X, Wc, bc);
    activations_pooled = reshape(pool(pool_dim, activations), [], m);
    
    probs = Wd * activations_pooled + repmat(bd, [1, m]);
    probs = exp(probs);
    probs = bsxfun(@rdivide, probs, sum(probs));
    
    y = y';
    L = (-1/m) * (y(:)'*log(probs(:)));
    
    delta_output = -(y - probs);
    delta_pool = reshape(Wd'*delta_output, p_size, p_size, filter_num, m);
    delta_conv = zeros(cov_dim, cov_dim, filter_num, m);
    parfor i = 1:m
        for j = 1:filter_num
            % mean pooling
%             delta_conv(:, :, j, i) = ...
%                 (1/pool_dim^2) * kron(delta_pool(:, :, j, i), ones(pool_dim));
            % max pooling
            delta_conv(:, :, j, i) = ...
                kron(delta_pool(:, :, j, i), ones(pool_dim));
        end
    end
    delta_conv = activations .* (1-activations) .* delta_conv;
    
    Wd_grad = 1/m .* delta_output * activations_pooled';
    bd_grad = 1/m .* sum(delta_output, 2);
    parfor i = 1:filter_num
        for j = 1:m
            Wc_grad(:, :, i) = Wc_grad(:, :, i) + ...
                conv2(X(:, :, j), rot90(delta_conv(:, :, i, j), 2), 'valid');
        end
        Wc_grad(:,:,i) = 1/m .* Wc_grad(:,:,i);

        temp = delta_conv(:,:,i,:);
        bc_grad(i) = 1/m .* sum(temp(:));
    end
    
    grad = [Wc_grad(:); Wd_grad(:); bc_grad(:); bd_grad(:)];
end

