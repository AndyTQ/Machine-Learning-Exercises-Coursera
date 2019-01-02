function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
results = []
possible_values = [0;0.01;0.03;0.1;0.3;1;3;10;30];
m = length(possible_values);
for i = 1:m
    for j = 1:m
        C_temp = possible_values(i);
        sigma_temp = possible_values(j);
        % your code goes here to train using C_test and sigma_test
        model = svmTrain(X, y, C_temp, @(x1, x2) ...
            gaussianKernel(x1, x2, sigma_temp)); 
        predictions = svmPredict(model, Xval);
        % and compute the validation set errors
        err_value = mean(double(predictions ~= yval));
        % save the results in the matrix
        results = [results ; C_temp sigma_temp err_value];
    end
end
% use the min() function to find the best C and sigma values
% select the column with the smallest error.
minimum_error = min(min(results(:,3)));
[p, ~]=find(results(:,3) == minimum_error);
C = results(p, 1);
sigma = results(p, 2);
% =========================================================================
end
