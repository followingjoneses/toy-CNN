m=input('input number of test images:');
image_dim = 28;
output_dim = 8;
X_test = zeros(image_dim, image_dim, m);

for i = 1:m
    fileName = strcat('test_data/', num2str(i), '.txt');
    a = textread(fileName);
    X_test(:, :, i) = a;
end

save testset X_test;