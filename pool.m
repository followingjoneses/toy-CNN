function pool_features = pool(pool_dim, cov_features)
    pool_filter = ones(pool_dim);

    m = size(cov_features, 4);
    filter_num = size(cov_features, 3);
    cov_dim = size(cov_features, 1);
    p_size = cov_dim/pool_dim;
    
    pool_features = zeros(p_size, p_size, filter_num, m);
    
    for i = 1:m
        for j = 1:filter_num
            feature = cov_features(:, :, j, i);
            % mean pooling
%             temp = conv2(feature, pool_filter, 'valid');
%             pooled = temp(1:pool_dim:end, 1:pool_dim:end) ./ (pool_dim * pool_dim);
%             pool_features(:, :, j, i) = pooled;
            % max pooling
            for k = 1:pool_dim:cov_dim-pool_dim+1
                for h = 1:pool_dim:cov_dim-pool_dim+1
                    tmp = feature(k:k+pool_dim-1, h:h+pool_dim-1);
                    max_value = max(max(tmp));
                    pool_features((k-1)/pool_dim+1, (h-1)/pool_dim+1, j, i) = max_value;
                end
            end
        end
    end
end

