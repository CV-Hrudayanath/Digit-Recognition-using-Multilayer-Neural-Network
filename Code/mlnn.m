function out = mlnn()
% Implementation of Multilayer Feedforward Neural Network with
% backpropagation learning.

% Network is trained and tested on MNIST database.  


tic
% Reading Image data

% Train Images and labels
train_images = loadMNISTImages('train-images.idx3-ubyte');
train_labels = loadMNISTLabels('train-labels.idx1-ubyte');

%Test Images and labels
test_images = loadMNISTImages('t10k-images.idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');



% FeedForward and Backpropagation Learning

% Parameters
learning_rate = 0.1;
theta = 0.000001;
no_of_input_units = 784;
no_of_hidden_units = 30;
no_of_output_units = 10;



% weights on input-hidden and hidden-output layer
input_to_hidden_weights = rand(no_of_hidden_units, no_of_input_units,1)/100;
hidden_to_output_weights = rand( no_of_output_units , no_of_hidden_units ,1)/100;

%net hidden and net output 
net_hidden = zeros(1,no_of_hidden_units);
net_output = zeros(1,no_of_output_units);

% y and z are outputs from sigmoid function in hidden and output layer
y = zeros(1,no_of_hidden_units);
z = zeros(1,no_of_output_units);


% sensitivity of hidden and output nodes
sensitivity_output = zeros(1,no_of_output_units);
sensitivity_hidden = zeros(1, no_of_hidden_units);



no_of_iterations = 0;     
while 1
       for patterns = 1:60000
            image = train_images(:,patterns);
            input = image;

            % Actual Result

            result = train_labels(patterns);
            t = zeros(1,10);
            if result == 0
               t(10) = 1;
            else
               t(result) = 1;
            end

            % calculation net hidden 
            for j  =  1:no_of_hidden_units
                sum = 0;
                for i = 1:no_of_input_units
                     sum = sum + input_to_hidden_weights(j,i)*input(i);
                end
                net_hidden(j) = sum;
            end


            % calculation of y
            for j = 1:no_of_hidden_units
                y(j) = sigmoid(net_hidden(j),0);
            end

            % calculation of net output 
            for k  =  1:no_of_output_units
                sum = 0;
                for j = 1:no_of_hidden_units
                     sum = sum + hidden_to_output_weights(k,j)*y(j);
                end
                net_output(k) = sum;
            end


            % calculation of z i.e the output of Neural Network 
            for k = 1:no_of_output_units
                z(k) = sigmoid(net_output(k),0);
            end


            
            % sensitivity for hidden and output layer

            for k = 1:no_of_output_units
                sensitivity_output(k) = (t(k) - z(k))*sigmoid(net_output(k),1);
            end

            for j = 1:no_of_hidden_units
                sum = 0;
                for k = 1:no_of_output_units
                    sum = sum + hidden_to_output_weights(k,j)*sensitivity_output(k);
                end
                sensitivity_hidden(j) = sigmoid(net_hidden(j),1)*sum;
            end


            
            % Backpropagation Learning
            
            % Input to hidden Weights update rule 
            for j = 1:no_of_hidden_units
                for i = 1:no_of_input_units
                     input_to_hidden_weights(j,i) = input_to_hidden_weights(j,i) + learning_rate*sensitivity_hidden(j)*input(i);
                    
                end
            end
            
            % Hidden to Output Weights update rule
            for k  = 1:no_of_output_units
                for j = 1:no_of_hidden_units
                    hidden_to_output_weights(k,j) = hidden_to_output_weights(k,j) + learning_rate*sensitivity_output(k)*y(j);
                end
            end
            
            %Error
            error = 0;
            for k = 1:no_of_output_units
                error = error + (t(k) - z(k))*(t(k)-z(k));
            end
            error = error/2;
            
           if error < theta
              break;
           end
           
    end
    
    

%Testing

count = 0;
confusion_matrix = zeros(10,10);

for test = 1:10000
            image = test_images(:,test);
            input = image;

            % Actual Result

            result = test_labels(test);
            t = zeros(1,10);
            if result == 0
               t(10) = 1;
            else
               t(result) = 1;
            end

             
            % Feedforward to Neural Network
            
            for j  =  1:no_of_hidden_units
                sum = 0;
                for i = 1:no_of_input_units
                     sum = sum + input_to_hidden_weights(j,i)*input(i);
                end
                net_hidden(j) = sum;
            end



            for j = 1:no_of_hidden_units
                y(j) = sigmoid(net_hidden(j),0);
            end


            for k  =  1:no_of_output_units
                sum = 0;
                for j = 1:no_of_hidden_units
                     sum = sum + hidden_to_output_weights(k,j)*y(j);
                end
                net_output(k) = sum;
            end


            % Z is obtained output from Network
            for k = 1:no_of_output_units
                z(k) = sigmoid(net_output(k),0);
            end
            
            
            % Finding the digit
            max = 0;
            for p = 1:no_of_output_units
                if z(p) > max 
                  max = z(p);
                  index = p;
                end
            end
            
            
            if result == 0
                result = 10;
            end
            
            
            confusion_matrix(result,index) = confusion_matrix(result,index) + 1;
            
            
            if index == result
                count = count + 1;
            end
            
            
            
end

% Performance

fprintf('No of Correctly Recognized Images: %d\n', count);
%disp(count);
fprintf('Confusion Matrix\n');
disp(confusion_matrix);
fprintf('Accuracy is %f\n',(count/10000)*100);


recall = zeros(1,10);
row_sum = zeros(1,10);
col_sum = zeros(1,10);

for i = 1:10
    for j = 1:10
       col_sum(i) = col_sum(i) + confusion_matrix(i,j);
    end
end

for j = 1:10
    for i = 1:10
       row_sum(j) = row_sum(j) + confusion_matrix(j,i);
    end
end



for i = 1:10
   recall(i) = confusion_matrix(i,i)/col_sum(i);   
end

precision = zeros(1,10);
for i = 1:10
   precision(i) = confusion_matrix(i,i)/row_sum(i);
end

fprintf('Recall is:\n');
disp(recall);

fprintf('Precision is:\n');
disp(precision);


if no_of_iterations > 6
       break;
end
%disp(no_of_iterations);
no_of_iterations = no_of_iterations +1;

toc

end