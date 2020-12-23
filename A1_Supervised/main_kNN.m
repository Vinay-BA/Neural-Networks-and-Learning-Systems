%% This script will help you test out your kNN code

%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 2; % Change this to load new data 

% X - Data samples
% D - Desired output from classifier for each sample
% L - Labels for each sample
[X, D, L] = loadDataSet( dataSetNr );

% You can plot and study dataset 1 to 3 by running:
plotCase(X,D);

%% Select a subset of the training samples

numBins = 5;                    % Number of bins you want to devide your data into
% change numBins to set the number of folds in cross validation
numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true;          % true = select samples at random, false = select the first features

[XBins, DBins, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom);

% Note: XBins, DBins, LBins will be cell arrays, to extract a single bin from them use e.g.
% XBin1 = XBins{1};
%
% Or use the combineBins helper function to combine several bins into one matrix (good for cross validataion)
% XBinComb = combineBins(XBins, [1,2]);

% Add your own code to setup data for training and test here
%X1 = XBins{1};
%X2 = XBins{2};
%X3 = XBins{3};
%L1 = LBins{1};
%L2 = LBins{2};
%L3 = LBins{3};

%% Use kNN to classify data
%  Note: you have to modify the kNN() function yourself.
maxK = 20; %set max value of k here
jVec = 1:numBins;
accTrain = zeros(length(jVec),1)';
accTest = zeros(length(jVec),1)';
accTrainK = zeros(maxK,1)';
accTestK = zeros(maxK,1)';

for i = 1:maxK
k = i; % Set the number of neighbors
for j = 1:length(jVec)
negjVec = [jVec(1:length(jVec) ~= j)];
XTest = XBins{jVec(j)};
XTrain = combineBins(XBins, negjVec);
LTest = LBins{jVec(j)};
LTrain = combineBins(LBins, negjVec);
    
% Classify training data
LPredTrain = kNN(XTrain, k, XTrain, LTrain);
% Classify test data
LPredTest = kNN(XTest , k, XTrain, LTrain);

% The confucionMatrix
cMTrain = calcConfusionMatrix(LPredTrain, LTrain);
cMTest = calcConfusionMatrix(LPredTest, LTest);

% The accuracy
accTrain(j) = calcAccuracy(cMTrain);
accTest(j) = calcAccuracy(cMTest);
end
accTrainK(i) = mean(accTrain);
accTestK(i) = mean(accTest);
end

[Mtrain,Itrain] = max(accTrainK); 
maxAccuTrain = Mtrain; %maximum accuracy on Training data 
bestKTrain = Itrain; %best k on Training data
[Mtest,Itest] = max(accTestK); 
maxAccuTest = Mtest; %maximum accuracy on Test data 
bestKTest = Itest; %best k on Test data

% plotting accuracy vs k-values
plot(1:1:maxK,accTrainK);
 hold on 
plot(1:1:maxK,accTestK);
 legend('Train Accuracy','Test Accuracy')
 title('Test/Train Accuracy vs k-values')
 hold off

%% Calculate The Confusion Matrix and the Accuracy
%  Note: you have to modify the calcConfusionMatrix() and calcAccuracy()
%  functions yourself.

% This part is included in the previous section

%% Plot classifications
%  Note: You should not have to modify this code

if dataSetNr < 4
    plotResultDots(XTrain, LTrain, LPredTrain, XTest, LTest, LPredTest, 'kNN', [], bestKTest);
else
    plotResultsOCR(XTest, LTest, LPredTest)
end
