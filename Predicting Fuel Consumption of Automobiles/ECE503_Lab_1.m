% 3.1 From the course website download D_mpg.mat
load /Users/adamcheung/Documents/Uvic/Code/ECE503/Lab_1/D_mpg.mat

% 3.2 Prepare matrix Xh_tr and y_tr
y = D_mpg(7,:)';
M = length(y);
P = 314;
T = M - P;
X = D_mpg(1:6,:);
Xh = [X;ones(1,M)];
Xh_tr = Xh(:,1:P);
y_tr = y(1:P);

% 3.3 Prepare test data X
Xh_te = Xh(:,P+1:M);
y_te = y(P+1:M);

% 3.4 Compute optimal parameter wh_star
wh_star = pinv((Xh_tr)') * y_tr;

% 3.5 Apply optimized model to the training and test data respectively
y_tr_pred = Xh_tr' * wh_star; % Optimized model on training data
y_te_pred = Xh_te' * wh_star; % Optimized model on testing data

RMSE_tr = sqrt(1/314 * (sum((y_tr_pred - y_tr).^2))); % Root-mean-sqaured error for training data
RMSE_te = sqrt(1/78 * (sum((y_te_pred - y_te).^2))); % Root-mean_sqaured error for testing data

% 3.6 plot y_te in blue and y_te_pred in red
figure;
plot(1:length(y_te), y_te, 'b-o', 'LineWidth', 1.5); hold on;
plot(1:length(y_te_pred), y_te_pred, 'r-*', 'LineWidth', 1.5);
legend('Ground Truth (y\_te)', 'Prediction (y\_te\_pred)');
xlabel('Test Samples {1,2,...78}');
ylabel('Miles Per Gallon (MPG)');
title('Ground Truth vs Prediction on Test Samples');
grid on;