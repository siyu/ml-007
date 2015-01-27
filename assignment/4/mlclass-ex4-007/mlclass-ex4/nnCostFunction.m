function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
				 
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% add 1s to the first column (bias input)
X = [ones(m,1) X];

% X is 5000x401, Theta1 is 25 x 401
z2 = X * Theta1';
a2 = sigmoid(z2);	% a2 is now 5000 x 25
a2 = [ones(size(a2,1),1) a2];	

% a2 is 5000x26, Theta is 10x26
z3 = a2 * Theta2';
a3 = sigmoid(z3);	% a3 is now 5000x10

a3_num_rows = size(a3,1);
a3_num_cols = size(a3,2);

error = zeros(a3_num_rows,1);

y_prime = zeros(size(y,1),a3_num_cols);
for i=1:size(y,1)
	y_prime(i,y(i))=1;
end

for i=1:a3_num_rows
	for j=1:a3_num_cols
		error(i) = error(i) .+ (-y_prime(i,j) .* log(a3(i,j)) .- (1 .- y_prime(i,j)) .* log(1 .- a3(i,j)));
	end
end

% no regularization
J = sum(error) ./ m ;

% regularization
Theta1_no_bias = Theta1(:,2:end);
Theta1_no_bias_sq = Theta1_no_bias .^ 2;
Theta2_no_bias = Theta2(:,2:end);
Theta2_no_bias_sq = Theta2_no_bias .^ 2;
reg = (sum(Theta1_no_bias_sq(:)) + sum(Theta2_no_bias_sq(:))) .* lambda ./ (2 .* m);

J = J .+ reg;

% Back propagation
% -------------------------------------------------------------
% part 1: see feed forward section above 
% part 2: calculate output layer error
d3 = a3 - y_prime;	% e3 is 5000 x 10

% part 3: calculate hidden layer error
d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2);	% e2 is 5000x25

% part 4: accumulate gradient
delta1 = d2' * X;	% 25 x 401
delta2 = d3' * a2;	% 10 x 26

Theta1_grad = delta1 ./ m;	% 25 x 401
Theta2_grad = delta2 ./ m;	% 10 x 26

% add regularization
Theta1_grad =  [Theta1_grad(:,1) (Theta1_grad(:,2:end) + lambda ./ m .* Theta1(:,2:end))];
Theta2_grad =  [Theta2_grad(:,1) (Theta2_grad(:,2:end) + lambda ./ m .* Theta2(:,2:end))];

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
