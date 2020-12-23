function [ cM ] = calcConfusionMatrix( LPred, LTrue )
% CALCCONFUSIONMATRIX returns the confusion matrix of the predicted labels

classes  = unique(LTrue);
NClasses = length(classes);

% Add your own code here
cM = zeros(NClasses);
for i = 1:NClasses  % true
    for j = 1:NClasses   % predicted
        cM(i,j) = sum((LPred == i)&(LTrue == j)); 
    end
end

end