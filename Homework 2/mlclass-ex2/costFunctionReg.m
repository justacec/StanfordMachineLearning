function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

theta = theta(:);
y = y(:);
x = X * theta;

J = (1.0 / length(y)) * sum((-1.0*y.*log(sigmoid(x))-((1-y).*log(1-sigmoid(x))))) + ((lambda/(2.0*length(y)))*(dot(theta, theta) - (theta(1)*theta(1))));
grad = cell2mat(arrayfun(@(n) {(1.0 / length(y)) * sum((sigmoid(x)-y).*X(:,n))}, 1:length(theta)));
reg = [0 cell2mat(arrayfun(@(n) {((lambda/length(y)) * n)}, theta(2:end)))'];
grad = grad + reg;

% =============================================================

end
