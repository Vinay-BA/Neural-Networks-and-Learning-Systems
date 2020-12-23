%% Hyper-parameters

% Number of randomized Haar-features
nbrHaarFeatures = 100;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 1000;
% Number of weak classifiers
nbrWeakClassifiers = 50;

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

%% Create image sets (do not modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError

Y = yTrain; 
D(1:length(Y),1) = 1/length(Y);
D = transpose(D);

for t = 1:nbrWeakClassifiers %looping over the number of weak classifiers
    Emin(t) = Inf;
    for j = 1:size(xTrain,1) %looping over each feature 
        X = xTrain(j,:); 
        P = 1;
        for T = X %looping over each element of the feature     
            C = WeakClassifier(T, P, X);
            E = WeakClassifierError(C, D, Y);
            if E > 0.5
                P = -P;
                E = 1-E;
                %C = -C; %could have done this instead of recomputing C
                %later but doing this will be computationally heavy.
            end
            if E < Emin(t)
                Emin(t) = E; %store all weak classifier's Error values
                Feature(t) = j; %store all weak classifier's Feature index values
                Threshold(t) = T; %store all weak classifier's Threshold values
                Polarity(t) = P; %store all weak classifier's Polarity values
            end      
        end
    end
    C = WeakClassifier(Threshold(t), Polarity(t), xTrain(Feature(t),:)); %indices for T P X
    alpha(t) = ((1/2)*log((1-Emin(t))/Emin(t)));
    D = D.*exp(-alpha(t)*(Y.*C)); 
    D = D/sum(D); %normalizing D
end

%% Evaluate your strong classifier here  
%  Evaluate on both the training data and test data, but only the test
%  accuracy can be used as a performance metric since the training accuracy
%  is biased.

for cn = 1:nbrWeakClassifiers %loop over all weak classifiers 
    trainClass(cn,:) = alpha(cn) * WeakClassifier(Threshold(cn), Polarity(cn), xTrain(Feature(cn),:)); 
    trainClassSumWeak(cn,:) =sum(trainClass,1); 
    predTrain(cn,(trainClassSumWeak(cn,:)>0)) = 1; 
    predTrain(cn,(trainClassSumWeak(cn,:)<0)) = -1;
    training(cn,(predTrain(cn,:) == yTrain)) = 0; %0 loss for right classification
    training(cn,(predTrain(cn,:) ~= yTrain)) = 1; %1 loss for wrong classification
    misclassificationsTrain(cn) = sum(training(cn,:));
    accuracyTrain(cn) = ((length(yTrain)-misclassificationsTrain(cn))/length(yTrain))*100;
    errorTrain(cn) = (misclassificationsTrain(cn)/length(yTrain))*100;
    
    testClass(cn,:) = alpha(cn) * WeakClassifier(Threshold(cn), Polarity(cn), xTest(Feature(cn),:));
    testClassSumWeak(cn,:) =sum(testClass,1);
    predTest(cn,(testClassSumWeak(cn,:)>0)) = 1; 
    predTest(cn,(testClassSumWeak(cn,:)<0)) = -1;
    testing(cn,(predTest(cn,:) == yTest)) = 0;
    testing(cn,(predTest(cn,:) ~= yTest)) = 1;
    misclassificationsTest(cn) = sum(testing(cn,:));
    accuracyTest(cn) = ((length(yTest)-misclassificationsTest(cn))/length(yTest))*100;
    errorTest(cn) = (misclassificationsTest(cn)/length(yTest))*100;
end


%accuracy and error on train data
accuracyTrain(50);
errorTrain(50);

%accuracy and error on test data
accuracyTest(50);
errorTest(50);
%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.

x = 1:nbrWeakClassifiers;

%Error plot for Train data
y = errorTrain;
plot(x,y);
title('Training error of the strong classifier as a function of the number of weak classifiers');
xlabel('Number of weak classifiers');
ylabel('Error');

%Error plot for Test data
y = errorTest;
plot(x,y);
title('Test error of the strong classifier as a function of the number of weak classifiers');
xlabel('Number of weak classifiers');
ylabel('Error');

%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.
 
indices = find(testing(size(testing,1),:)); %indices of misclassified samples
colormap gray;
for k = 1:25
    subplot(5,5,k), imagesc(testImages(:,:,indices(k)));
    axis image;
    axis off;
end

%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.

colormap gray;
for k = 1:length(Feature)
    subplot(7,8,k),imagesc(haarFeatureMasks(:,:,Feature(k)));
    axis image;
    axis off;
end
