function out1 = k_nn()
% 1-Nearest Neighbours method for recognizing digits. 

% Reading Image data
train_images = loadMNISTImages('train-images.idx3-ubyte');
train_labels = loadMNISTLabels('train-labels.idx1-ubyte');

test_images = loadMNISTImages('t10k-images.idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

count = 0;
tic
for i = 1:1500
    image1 = test_images(:,i);
    image1 = image1';
    min = 10000;
    for j = 1:10000
       image2 = train_images(:,j);
       image2 = image2';
       distance = pdist2(image1,image2);
       if distance < min
           min = distance;
           min_index = j;
       end
       
    end
    disp(i);
    disp(train_labels(min_index));
    disp(test_labels(i));
    if test_labels(i) == train_labels(min_index)
       count = count + 1;
    end
    
    
end

disp(count);

toc

end

