%% 4.1 From the course website download X_iris.mat and add the LRBC folder to the Matlab path
load /Users/adamcheung/Documents/Uvic/Code/ECE503/Lab_2/X_iris.mat
addpath('/Users/adamcheung/Documents/Uvic/Code/ECE503/Lab_2/LRBC');

%% 4.2 Follow Section 3 to prepare training and test data
X1 = X_iris (:, 1:50);
Xtr1 = X1 (:, 1:35);
Xte1 = X1 (:, 36:50);
X2 = X_iris (:, 51:100);
Xtr2 = X2 (:, 1:35);
Xte2 = X2 (:, 36:50);
X3 = X_iris (:, 101:150);
Xtr3 = X3 (:, 1:35);
Xte3 = X3 (:, 36:50);


%% 4.3 Prepare code that applies Step 1 and 2 of the One-Versus-All 3-class algo
% Input: Prepare training data input for One-Versus_All 3-class Algo
D_setosa_tr = Xtr1;
D_not_setosa_tr = [Xtr2 Xtr3];

D_versicolor_tr = Xtr2;
D_not_versicolor_tr = [Xtr1 Xtr3];

D_virginica_tr = Xtr3;
D_not_virginica_tr = [Xtr1 Xtr2];

X_setosa_tr = [D_setosa_tr D_not_setosa_tr];
X_versicolor_tr = [D_versicolor_tr D_not_versicolor_tr];
X_viginica_tr = [D_virginica_tr D_not_virginica_tr];

y = [ones(1,35) -ones(1,70)];

% Step 1: Apply binary classification to each One-Versus-All pair
[ws_setosa,C2_setosa] = LRBC_newton(X_setosa_tr,y,5);
[ws_versicolor,C2_versicolor] = LRBC_newton(X_versicolor_tr,y,5);
[ws_virginica,C2_virginica] = LRBC_newton(X_viginica_tr,y,5);

ws1 = ws_setosa(1:end-1);
ws2 = ws_versicolor(1:end-1);
ws3 = ws_virginica(1:end-1);

% Step 2: Normalize the 3 paris of parameters obatined from Step 1
ws_setosa_norm = ws_setosa/norm(ws1);
ws_versicolor_norm = ws_versicolor/norm(ws2);
ws_virginica_norm = ws_virginica/norm(ws3);

% Step 3: Assigning the sample to the class with the largest decision score
function predicted_labels = classifyByDot(X, ws_setosa_norm, ws_versicolor_norm, ws_virginica_norm)
    % Compute the decision scores for each column of X against the three classifiers
    scores = [ws_setosa_norm' * X;
              ws_versicolor_norm' * X;
              ws_virginica_norm' * X];

    % Select the classifier with the largest score for each column (1=setosa, 2=versicolor, 3=virginica)
    [~, predicted_labels] = max(scores, [], 1);

    % Return the labels as a 15x1 column vector
    predicted_labels = predicted_labels.';
end


%% 4.4 Apply the classifier to training set for K = 5 Newton iteration and report confusion matrix and classification accuracy
Xtr1_h = [Xtr1; ones(1,35)];
Xtr2_h = [Xtr2; ones(1,35)];
Xtr3_h = [Xtr3; ones(1,35)];

y_tr_pred_1 = classifyByDot(Xtr1_h,ws_setosa_norm, ws_versicolor_norm, ws_virginica_norm); %% Classify training samples from class 1
y_tr_pred_2 = classifyByDot(Xtr2_h,ws_setosa_norm, ws_versicolor_norm, ws_virginica_norm); %% Classify training samples from class 2
y_tr_pred_3 = classifyByDot(Xtr3_h,ws_setosa_norm, ws_versicolor_norm, ws_virginica_norm); %% Classify training samples from class 3

y_tr_pred = [y_tr_pred_1' y_tr_pred_2' y_tr_pred_3']; %% Combine all predicted labels
y_tr_true = [ones(1,35), 2*ones(1,35), 3*ones(1,35)]; %% Combine all true labels

% Compute confusion matrix
C3_tr_k5 = zeros(3,3);
for i = 1:length(y_tr_true)
    GroundTruth = y_tr_true(i);
    PredClass = y_tr_pred(i);
    C3_tr_k5(PredClass, GroundTruth) = C3_tr_k5(PredClass, GroundTruth)+1;
end
disp(C3_tr_k5);

% Compute overall classification accuracy
Accuracy_tr_k5 = trace(C3_tr_k5) / sum(C3_tr_k5(:));
disp(Accuracy_tr_k5);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Replicate results for K = 1 Newton iteration on training data
[ws_setosa,C2_setosa] = LRBC_newton(X_setosa_tr,y,1);
[ws_versicolor,C2_versicolor] = LRBC_newton(X_versicolor_tr,y,1);
[ws_virginica,C2_virginica] = LRBC_newton(X_viginica_tr,y,1);

ws1 = ws_setosa(1:end-1);
ws2 = ws_versicolor(1:end-1);
ws3 = ws_virginica(1:end-1);

ws_setosa_norm = ws_setosa/norm(ws1);
ws_versicolor_norm = ws_versicolor/norm(ws2);
ws_virginica_norm = ws_virginica/norm(ws3);

Xtr1_h = [Xtr1; ones(1,35)];
Xtr2_h = [Xtr2; ones(1,35)];
Xtr3_h = [Xtr3; ones(1,35)];

y_tr_pred_1 = classifyByDot(Xtr1_h,ws_setosa_norm, ws_versicolor_norm, ws_virginica_norm); %% Classify training samples from class 1
y_tr_pred_2 = classifyByDot(Xtr2_h,ws_setosa_norm, ws_versicolor_norm, ws_virginica_norm); %% Classify training samples from class 2
y_tr_pred_3 = classifyByDot(Xtr3_h,ws_setosa_norm, ws_versicolor_norm, ws_virginica_norm); %% Classify training samples from class 3

y_tr_pred = [y_tr_pred_1' y_tr_pred_2' y_tr_pred_3']; 
y_tr_true = [ones(1,35), 2*ones(1,35), 3*ones(1,35)]; 

% Confusion Matrix
C3_tr_k1 = zeros(3,3);
for i = 1:length(y_tr_true)
    GroundTruth = y_tr_true(i);
    PredClass = y_tr_pred(i);
    C3_tr_k1(PredClass, GroundTruth) = C3_tr_k1(PredClass, GroundTruth)+1;
end
disp(C3_tr_k1);

% Accuracy
Accuracy_tr_k1 = trace(C3_tr_k1) / sum(C3_tr_k1(:));
disp(Accuracy_tr_k1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Replicate results for K = 3 Newton iteration on training data
[ws_setosa,C2_setosa] = LRBC_newton(X_setosa_tr,y,3);
[ws_versicolor,C2_versicolor] = LRBC_newton(X_versicolor_tr,y,3);
[ws_virginica,C2_virginica] = LRBC_newton(X_viginica_tr,y,3);

ws1 = ws_setosa(1:end-1);
ws2 = ws_versicolor(1:end-1);
ws3 = ws_virginica(1:end-1);

ws_setosa_norm = ws_setosa/norm(ws1);
ws_versicolor_norm = ws_versicolor/norm(ws2);
ws_virginica_norm = ws_virginica/norm(ws3);

y_tr_pred_1 = classifyByDot(Xtr1_h,ws_setosa_norm, ws_versicolor_norm, ws_virginica_norm); %% Classify training samples from class 1
y_tr_pred_2 = classifyByDot(Xtr2_h,ws_setosa_norm, ws_versicolor_norm, ws_virginica_norm); %% Classify training samples from class 2
y_tr_pred_3 = classifyByDot(Xtr3_h,ws_setosa_norm, ws_versicolor_norm, ws_virginica_norm); %% Classify training samples from class 3

y_tr_pred = [y_tr_pred_1' y_tr_pred_2' y_tr_pred_3']; 
y_tr_true = [ones(1,35), 2*ones(1,35), 3*ones(1,35)];

% Confusion Matrix
C3_tr_k3 = zeros(3,3);
for i = 1:length(y_tr_true)
    GroundTruth = y_tr_true(i);
    PredClass = y_tr_pred(i);
    C3_tr_k3(PredClass, GroundTruth) = C3_tr_k3(PredClass, GroundTruth)+1;
end
disp(C3_tr_k3);

% Accuracy
Accuracy_tr_k3 = trace(C3_tr_k3) / sum(C3_tr_k3(:));
disp(Accuracy_tr_k3);


%% 4.4 Apply the classifier to testing set for K = 5 Newton iteration and report confusion matrix and classification accuracy
[ws_setosa,C2_setosa] = LRBC_newton(X_setosa_tr,y,5);
[ws_versicolor,C2_versicolor] = LRBC_newton(X_versicolor_tr,y,5);
[ws_virginica,C2_virginica] = LRBC_newton(X_viginica_tr,y,5);

ws1 = ws_setosa(1:end-1);
ws2 = ws_versicolor(1:end-1);
ws3 = ws_virginica(1:end-1);

ws_setosa_norm = ws_setosa/norm(ws1);
ws_versicolor_norm = ws_versicolor/norm(ws2);
ws_virginica_norm = ws_virginica/norm(ws3);

Xte1_h = [Xte1; ones(1,15)];
Xte2_h = [Xte2; ones(1,15)];
Xte3_h = [Xte3; ones(1,15)];

y_te_pred_1 = classifyByDot(Xte1_h,ws_setosa_norm, ws_versicolor_norm, ws_virginica_norm); %% Classify testing samples from class 1
y_te_pred_2 = classifyByDot(Xte2_h,ws_setosa_norm, ws_versicolor_norm, ws_virginica_norm); %% Classify testing samples from class 2
y_te_pred_3 = classifyByDot(Xte3_h,ws_setosa_norm, ws_versicolor_norm, ws_virginica_norm); %% Classify testing samples from class 3

y_te_pred = [y_te_pred_1' y_te_pred_2' y_te_pred_3']; 
y_te_true = [ones(1,15), 2*ones(1,15), 3*ones(1,15)];

% Confusion Matrix
C3_te_k5 = zeros(3,3);
for i = 1:length(y_te_true)
    GroundTruth = y_te_true(i);
    PredClass = y_te_pred(i);
    C3_te_k5(PredClass, GroundTruth) = C3_te_k5(PredClass, GroundTruth)+1;
end
disp(C3_te_k5);

% Accuracy
Accuracy_te_k5 = trace(C3_te_k5) / sum(C3_te_k5(:));
disp(Accuracy_te_k5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Replicate results for K = 1 Newton iteration on testing data
[ws_setosa,C2_setosa] = LRBC_newton(X_setosa_tr,y,1);
[ws_versicolor,C2_versicolor] = LRBC_newton(X_versicolor_tr,y,1);
[ws_virginica,C2_virginica] = LRBC_newton(X_viginica_tr,y,1);

ws1 = ws_setosa(1:end-1);
ws2 = ws_versicolor(1:end-1);
ws3 = ws_virginica(1:end-1);

ws_setosa_norm = ws_setosa/norm(ws1);
ws_versicolor_norm = ws_versicolor/norm(ws2);
ws_virginica_norm = ws_virginica/norm(ws3);

y_te_pred_1 = classifyByDot(Xte1_h,ws_setosa_norm, ws_versicolor_norm, ws_virginica_norm); %% Classify testing samples from class 1
y_te_pred_2 = classifyByDot(Xte2_h,ws_setosa_norm, ws_versicolor_norm, ws_virginica_norm); %% Classify testing samples from class 2
y_te_pred_3 = classifyByDot(Xte3_h,ws_setosa_norm, ws_versicolor_norm, ws_virginica_norm); %% Classify testing samples from class 3

y_te_pred = [y_te_pred_1' y_te_pred_2' y_te_pred_3']; 
y_te_true = [ones(1,15), 2*ones(1,15), 3*ones(1,15)];

% Confusion Matrix
C3_te_k1 = zeros(3,3);
for i = 1:length(y_te_true)
    GroundTruth = y_te_true(i);
    PredClass = y_te_pred(i);
    C3_te_k1(PredClass, GroundTruth) = C3_te_k1(PredClass, GroundTruth)+1;
end
disp(C3_te_k1);

% Accuracy
Accuracy_te_k1 = trace(C3_te_k1) / sum(C3_te_k1(:));
disp(Accuracy_te_k1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Replicate results for K = 3 Newton iteration on testing data
[ws_setosa,C2_setosa] = LRBC_newton(X_setosa_tr,y,3);
[ws_versicolor,C2_versicolor] = LRBC_newton(X_versicolor_tr,y,3);
[ws_virginica,C2_virginica] = LRBC_newton(X_viginica_tr,y,3);

ws1 = ws_setosa(1:end-1);
ws2 = ws_versicolor(1:end-1);
ws3 = ws_virginica(1:end-1);

ws_setosa_norm = ws_setosa/norm(ws1);
ws_versicolor_norm = ws_versicolor/norm(ws2);
ws_virginica_norm = ws_virginica/norm(ws3);

y_te_pred_1 = classifyByDot(Xte1_h,ws_setosa_norm, ws_versicolor_norm, ws_virginica_norm); %% Classify testing samples from class 1
y_te_pred_2 = classifyByDot(Xte2_h,ws_setosa_norm, ws_versicolor_norm, ws_virginica_norm); %% Classify testing samples from class 2
y_te_pred_3 = classifyByDot(Xte3_h,ws_setosa_norm, ws_versicolor_norm, ws_virginica_norm); %% Classify testing samples from class 3

y_te_pred = [y_te_pred_1' y_te_pred_2' y_te_pred_3']; 
y_te_true = [ones(1,15), 2*ones(1,15), 3*ones(1,15)];

% Confusion Matrix
C3_te_k3 = zeros(3,3);
for i = 1:length(y_te_true)
    GroundTruth = y_te_true(i);
    PredClass = y_te_pred(i);
    C3_te_k3(PredClass, GroundTruth) = C3_te_k3(PredClass, GroundTruth)+1;
end
disp(C3_te_k3);

% Accuracy
Accuracy_te_k3 = trace(C3_te_k3) / sum(C3_te_k3(:));
disp(Accuracy_te_k3);
