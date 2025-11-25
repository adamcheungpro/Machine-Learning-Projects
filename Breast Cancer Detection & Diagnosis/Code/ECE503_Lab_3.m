%% 4.1 Download and load data matrix
addpath('/Users/adamcheung/Documents/Uvic/Code/ECE503/Lab_3');
load /Users/adamcheung/Documents/Uvic/Code/ECE503/Lab_3/D_wdbc.mat

%% 4.2 Gnerate normalized training and test data sets and their labels

% Preparing the Data
Dtr = D_wdbc(:,1:285);
Dte = D_wdbc(:,286:569);

% Normalize the Data Sets
m = zeros(1,30);
v = zeros(1,30);

Xtr = zeros(30,285); % 30 features, 285 training samples, (30 * 285)
for i = 1:30
    xi = Dtr(i,:);
    m(i) = mean(xi);
    v(i) = sqrt(var(xi));
    Xtr(i,:) = (xi - m(i)) / v(i);
end

Xte = zeros(30,284); % 30 features, 284 test samples, (30 * 284)
for i = 1:30
    xi = Dte(i,:);
    Xte(i,:) = (xi - m(i))/v(i);
end

ytr = Dtr(31,:); % 285 Training Labels, (1 * 285)
yte = Dte(31,:); % 284 Test Labels, (1 * 284)

Dtr_hat = [Xtr; ytr];

%% 4.3 Prepare functions for evaluating the regularized cost function and its gradient
% Refer to f_wdbc.m and g_wdbc created for calculating regularized softmax cost function and its gradient

%% 4.4 Modify grad_desc.m according to the Remark

%% 4.5 Set the initial point w0 = 0 and tun the code to obtain optimized parameters for the following settings

% i) w0 = 0 "zeros(31,1)", u = 0 and K = 10
[ws1_GD, ELRs1_GD, k1_GD] = grad_desc('f_wdbc','g_wdbc',zeros(31,1), Dtr_hat, 0, 10);

% ii) w0 = 0 "zeros(31,1)", u = 0.1 and K = 10
[ws2_GD, ELRs2_GD, k2_GD] = grad_desc('f_wdbc','g_wdbc',zeros(31,1), Dtr_hat, 0.1, 10);

% iii) w0 = 0 "zeros(31,1)", u = 0 and K = 30
[ws3_GD, ELRs3_GD, k3_GD] = grad_desc('f_wdbc','g_wdbc',zeros(31,1), Dtr_hat, 0, 30);

% iv) w0 = 0 "zeros(31,1)", u = 0.075 and K = 30
[ws4_GD, ELRs4_GD, k4_GD] = grad_desc('f_wdbc','g_wdbc',zeros(31,1), Dtr_hat, 0.075, 30);

%% 4.6 Classify the 284 test samples using the optimized models above
Xte_hat = [Xte; ones(1,size(Xte,2))];

% Predict test sample using model i) and report confusion matrix & accuracy
y_pred_1 = sign(ws1_GD' * Xte_hat);

labelToIdx = @(lbl) (lbl + 3) / 2;      % -1 -> 1, +1 -> 2
C1_te = zeros(2, 2);
for i = 1:numel(y_pred_1)
    gtIdx   = labelToIdx(yte(i));
    predIdx = labelToIdx(y_pred_1(i));
    C1_te(predIdx, gtIdx) = C1_te(predIdx, gtIdx) + 1;
end
disp(C1_te);

Accuracy_te_1 = trace(C1_te) / sum(C1_te(:));
disp(Accuracy_te_1);

% Predict test sample using model ii) and report confusion matrix & accuracy
y_pred_2 = sign(ws2_GD' * Xte_hat);

labelToIdx = @(lbl) (lbl + 3) / 2;      % -1 -> 1, +1 -> 2
C2_te = zeros(2, 2);
for i = 1:numel(y_pred_2)
    gtIdx   = labelToIdx(yte(i));
    predIdx = labelToIdx(y_pred_2(i));
    C2_te(predIdx, gtIdx) = C2_te(predIdx, gtIdx) + 1;
end
disp(C2_te);

Accuracy_te_2 = trace(C2_te) / sum(C2_te(:));
disp(Accuracy_te_2);

% Predict test sample using model iii) and report confusion matrix & accuracy
y_pred_3 = sign(ws3_GD' * Xte_hat);

labelToIdx = @(lbl) (lbl + 3) / 2;      % -1 -> 1, +1 -> 2
C3_te = zeros(2, 2);
for i = 1:numel(y_pred_3)
    gtIdx   = labelToIdx(yte(i));
    predIdx = labelToIdx(y_pred_3(i));
    C3_te(predIdx, gtIdx) = C3_te(predIdx, gtIdx) + 1;
end
disp(C3_te);

Accuracy_te_3 = trace(C3_te) / sum(C3_te(:));
disp(Accuracy_te_3);

% Predict test sample using model iv) and report confusion matrix & accuracy
y_pred_4 = sign(ws4_GD' * Xte_hat);

labelToIdx = @(lbl) (lbl + 3) / 2;      % -1 -> 1, +1 -> 2
C4_te = zeros(2, 2);
for i = 1:numel(y_pred_4)
    gtIdx   = labelToIdx(yte(i));
    predIdx = labelToIdx(y_pred_4(i));
    C4_te(predIdx, gtIdx) = C4_te(predIdx, gtIdx) + 1;
end
disp(C4_te);

Accuracy_te_4 = trace(C4_te) / sum(C4_te(:));
disp(Accuracy_te_4);