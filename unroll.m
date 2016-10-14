function [Wc, Wd, bc, bd] = unroll(params, filter_dim, filter_num, hidden_size, output_dim)
    s = 1;
    e = filter_dim^2*filter_num;
    Wc = reshape(params(s:e), filter_dim, filter_dim, filter_num);
    s = e+1;
    e = e+hidden_size*output_dim;
    Wd = reshape(params(s:e), output_dim, hidden_size);
    s = e+1;
    e = e+filter_num;
    bc = params(s:e);
    bd = params(e+1:end);
end

