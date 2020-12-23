function [ LPred ] = kNN(X, k, XTrain, LTrain)
% KNN Your implementation of the kNN algorithm
%    Inputs:
%              X      - Samples to be classified (matrix)
%              k      - Number of neighbors (scalar)
%              XTrain - Training samples (matrix)
%              LTrain - Correct labels of each sample (vector)
%
%    Output:
%              LPred  - Predicted labels for each sample (vector)

%classes = unique(LTrain);
%NClasses = length(classes);

% Add your own code here
%LPred = zeros(size(X,1),1);

distmat = pdist2(X, XTrain); %distance matrix
%distmat = transpose(distmat);
[m,i] = mink(distmat,k,2); %applying mink rowwise

classlabel = LTrain(i);
LPred = mode(classlabel,2);
%LPred = transpose(LPred);
end

