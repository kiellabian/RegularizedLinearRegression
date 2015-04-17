function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

Hypothesis = X*theta;
%Hypothesis = (X * theta(2)) + theta(1)
difference = Hypothesis .- y;
diff_squared = difference .^ 2;
diff_squared = sum(diff_squared);
J = diff_squared / (2 * m);
theta = [0; theta(2:end, :);];
sumOfThetas = sum(theta .^ 2);
sumOfThetas = sumOfThetas * lambda / (2 * m);
J = J + sumOfThetas;


%grad2 = (1/m)*(X'*difference + lambda*theta(2))
  
%h = X*theta;

% regularize theta by removing first value
%theta_reg = [0;theta(2:end, :);];
%J = (1/(2*m))*sum((h-y).^2)+(lambda/(2*m))*theta_reg'*theta_reg;

%grad = (1/m)*(X'*(h-y)+lambda*theta_reg);



% =========================================================================

for j=1:size(theta)
  grad(j) = (1/m)*((Hypothesis .- y)'*X(:,j)) + (lambda/m)*theta(j);
end

end
