% training dataset
files = dir(strcat('training_data/', '*.txt'));

m = length(files);
image_dim = 28;
output_dim = 8;
X_train = zeros(image_dim, image_dim, m);
y_train = zeros(m, output_dim);

for i = 1:m
    fileName = strcat('training_data/', files(i).name);
    a = textread(fileName, '' , 'headerlines', 1);
    X_train(:, :, i) = a;

    fid = fopen(fileName, 'r');
    b = fgetl(fid);
    y_line = zeros(1, output_dim);
    switch(b)
        case 'A'
            y_line = [1 0 0 0 0 0 0 0];
        case 'B'
            y_line = [0 1 0 0 0 0 0 0];
        case 'C'
            y_line = [0 0 1 0 0 0 0 0];
        case 'D'
            y_line = [0 0 0 1 0 0 0 0];
        case 'E'
            y_line = [0 0 0 0 1 0 0 0];
        case 'F'
            y_line = [0 0 0 0 0 1 0 0];
        case 'G'
            y_line = [0 0 0 0 0 0 1 0];
        case 'H'
            y_line = [0 0 0 0 0 0 0 1];
    end
    y_train(i, :) = y_line;
    fclose(fid);
end

save dataset X_train y_train;

% validation dataset
files = dir(strcat('validation_data/', '*.txt'));

m = length(files);
image_dim = 28;
output_dim = 8;
X_val = zeros(image_dim, image_dim, m);
y_val = zeros(m, output_dim);

for i = 1:m
    fileName = strcat('validation_data/', files(i).name);
    a = textread(fileName, '' , 'headerlines', 1);
    X_val(:, :, i) = a;

    fid = fopen(fileName, 'r');
    b = fgetl(fid);
    y_line = zeros(1, output_dim);
    switch(b)
        case 'A'
            y_line = [1 0 0 0 0 0 0 0];
        case 'B'
            y_line = [0 1 0 0 0 0 0 0];
        case 'C'
            y_line = [0 0 1 0 0 0 0 0];
        case 'D'
            y_line = [0 0 0 1 0 0 0 0];
        case 'E'
            y_line = [0 0 0 0 1 0 0 0];
        case 'F'
            y_line = [0 0 0 0 0 1 0 0];
        case 'G'
            y_line = [0 0 0 0 0 0 1 0];
        case 'H'
            y_line = [0 0 0 0 0 0 0 1];
    end
    y_val(i, :) = y_line;
    fclose(fid);
end

save validation X_val y_val;