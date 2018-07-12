% input data
input_filename = 'test_data/six.png';
input_image = imread(input_filename);
input_image = uint8(input_image);
figure(1);
image(input_image); axis image; axis off; colormap(gray(256));

% normalise image
input_image = double(input_image);
input_image = input_image / 255;

% read weights and biases
conv1_w = read_array('model/conv1.0.bin',5,5,1,20);
conv1_b = read_array('model/conv1.1.bin',20,1,1,1);

conv2_w = read_array('model/conv2.0.bin',5,5,20,50);
conv2_b = read_array('model/conv2.1.bin',50,1,1,1);

ip1_w = read_array('model/ip1.0.bin',800,1,1,500);
ip1_b = read_array('model/ip1.1.bin',500,1,1,1);

ip2_w = read_array('model/ip2.0.bin',500,1,1,10);
ip2_b = read_array('model/ip2.1.bin',10,1,1,1);

conv1_w = reformat_weight(conv1_w);
conv2_w = reformat_weight(conv2_w);
data = input_image;


data_conv1 = convolution_full( data, conv1_w, conv1_b, 0 ); % conv1
conv1_pool1 = max_pool( data_conv1 ); % pool1
pool1_conv2 = convolution_full(conv1_pool1,conv2_w, conv2_b, 0); % conv2
conv2_pool2 = max_pool(pool1_conv2); % pool2
conv2_pool2_v = vectorise_tensor( conv2_pool2 );

% ip1 (inner product)
ip1_w_relu1_in = sum(repmat(conv2_pool2_v,500,1) .* ip1_w',2)';
ip1_w_relu1_in = ip1_w_relu1_in + ip1_b';
ip1_w_relu1 = ip1_w_relu1_in;
ip1_w_relu1(ip1_w_relu1_in < 0) = 0; % relu1
% ipl2
relu1_ip2_w = sum(repmat(ip1_w_relu1,10,1) .* ip2_w',2)';
relu1_ip2_w = relu1_ip2_w + ip2_b';
relu1_ip2_w = relu1_ip2_w ./ max(relu1_ip2_w);
% prob
sm_exp = exp(relu1_ip2_w);
sm = sm_exp ./ sum(sm_exp);
digit = find(sm == max(sm)) - 1;
msg = sprintf('digit=%d',digit)








