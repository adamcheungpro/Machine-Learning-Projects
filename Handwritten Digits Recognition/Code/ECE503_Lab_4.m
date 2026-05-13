%% 5.1 Download and load data matrices
addpath('/Users/adamcheung/Documents/Uvic/Code/ECE503/Lab_4');
load /Users/adamcheung/Documents/Uvic/Code/ECE503/Lab_4/X1600.mat
load /Users/adamcheung/Documents/Uvic/Code/ECE503/Lab_4/Te28.mat
load /Users/adamcheung/Documents/Uvic/Code/ECE503/Lab_4/Lte28.mat

%% 5.2  Prepare  two  pairs  of  input  data  matrices,  namely {Dtr, Dte} and {Dhtr, Dhte}
u = ones(1,1600);
ytr = [u 2*u 3*u 4*u 5*u 6*u 7*u 8*u 9*u 10*u]; 
Dtr = [X1600; ytr]; % Raw training data matrix with 784 raw features and 1 bias (785 * 16,000)
Dte = [Te28; 1+Lte28(:)']; % Raw testing data matrix with 784 raw features and 1 bias (785 * 10,000)

H = [];
for i = 1:16000
    xi = X1600(:,i);
    mi = reshape(xi,28,28);
    hi = hog20(mi,7,9);
    H = [H hi];
end

Hte = [];
for i = 1:length(Lte28)
    xi = Te28(:,i);
    mi = reshape(xi,28,28);
    hi = hog20(mi,7,9);
    Hte = [Hte hi];
end

Dhtr = [H; ytr]; % HOG training data matrix with 576 HOG features and 1 bias (577 * 16,000)
Dhte = [Hte; 1+Lte28(:)']; % HOG testing data matrix with 576 HOG features and 1 bias (577 * 10,000)

%% 5.3 Apply SRMCC to the original datasets to classify the test samples
addpath('/Users/adamcheung/Documents/Uvic/Code/ECE503/Lab_4/SRMCC');

[Ws, f] = SRMCC_bfgsML(Dtr, "f_SRMCC", "g_SRMCC", 0.002, 10, 62); % Recommended mu = 0.002, K = 10 and iter = 62;

% Predict the labels for test samples
Dtest = [Dte(1:end-1, :); ones(1, size(Dte, 2))]; % Add bias row to test data matrix
tic; % Measure recognition time for raw features
scores = Dtest' * Ws;
[~, ypred] = max(scores, [], 2);
raw_recognition_time = toc;

% Construct 10*10 confusion matrix
C_te = zeros(10,10);
ytrue = Dte(end, :);
for i = 1:numel(ypred)
    predIdx = ypred(i);
    gtIdx = ytrue(i);
    C_te(predIdx, gtIdx) = C_te(predIdx, gtIdx) + 1;
end
disp("Confusion Matrix for raw features testing dataset:")
disp(C_te);

% Compute prediction accuracy
accuracy_te = trace(C_te) / sum(C_te(:));
disp("Prediction Accuracy for raw featues testing dataset:")
disp(accuracy_te);
raw_digits_per_second = numel(ypred) / raw_recognition_time;

% Return recognition speed (efficiency)
disp("Recognition speed (raw features) - digits per second:")
disp(raw_digits_per_second);

%% 5.4 Apply SRMCC to the HOG feature datasets to classify the test samples
addpath('/Users/adamcheung/Documents/Uvic/Code/ECE503/Lab_4/SRMCC');

[Ws_hog, f_hog] = SRMCC_bfgsML(Dhtr, "f_SRMCC", "g_SRMCC", 0.001, 10, 57); % Recommended mu = 0.001, K = 10 and iter = 57;

% Predict the labels for hog test samples
Dhtest = [Dhte(1:end-1, :); ones(1, size(Dhte, 2))]; % Add bias row to test data matrix
tic; % Measure recognition time for HOG features
scores = Dhtest' * Ws_hog;
[~, ypred_hog] = max(scores, [], 2);
hog_recognition_time = toc;

% Construct 10*10 confusion matrix
C_te_hog = zeros(10,10);
ytrue = Dhte(end, :);
for i = 1:numel(ypred_hog)
    predIdx = ypred_hog(i);
    gtIdx = ytrue(i);
    C_te_hog(predIdx, gtIdx) = C_te_hog(predIdx, gtIdx) + 1;
end
disp("Confusion Matrix for HOG features testing dataset:")
disp(C_te_hog);

% Compute prediction accuracy
accuracy_te_hog = trace(C_te_hog) / sum(C_te_hog(:));
disp("Prediction Accuracy for HOG featues testing dataset:")
disp(accuracy_te_hog);
hog_digits_per_second = numel(ypred_hog) / hog_recognition_time;

% Return recognition speed (efficiency)
disp("Recognition speed (HOG features) - digits per second:")
disp(hog_digits_per_second);

