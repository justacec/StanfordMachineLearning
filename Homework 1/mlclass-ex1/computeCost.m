function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

theta = theta(:);
y = y(:);

% Find out if the X variable is in the right orientation
x1r = X(1,:);
x1c = X(:,1);

x1r = x1r(:);
x1c = x1c(:);

is_x1r_1s = sum(x1r == 1) == length(x1r);
is_x1c_1s = sum(x1c == 1) == length(x1c);

if(is_x1r_1s == false && is_x1c_1s == false)
    error('The X matrix does not seem to include the intercept term\n')
end

if(is_x1r_1s == true)
    X = X';
end

p = X * theta;

J = sum((p - y).^2) * (1/(2*length(y)));

% =========================================================================

end
