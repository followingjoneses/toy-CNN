function cov_features = convolve(X, filters, bias)
    m = size(X, 3);
    image_dim = size(X, 1);
    filter_num = size(filters, 3);
    filter_dim = size(filters, 1);
    
    cov_dim = image_dim - filter_dim +1;
    
    cov_features = zeros(cov_dim, cov_dim, filter_num, m);
    
    for i = 1:m
        image = X(:, :, i);
        for j = 1:filter_num
            cov_image = zeros(cov_dim, cov_dim);
            
            filter = filters(:, :, j);
            filter = rot90(filter, 2);
            
            cov_image = conv2(image, filter, 'valid');
            cov_image = sigmoid(cov_image + bias(j));
            
            cov_features(:, :, j, i) = cov_image;
        end
    end
end

