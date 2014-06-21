function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

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

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    theta_tmp = zeros(length(theta), 1);
    
    for i = 1:length(theta)
        d = dot(((X * theta) - y), X(:,i)) / m;
        theta_tmp(i) = theta(i) - alpha * d;
    end

    theta = theta_tmp;

    fprintf('Theta: %10.4f   %10.4f\n', theta(1), theta(2))

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
