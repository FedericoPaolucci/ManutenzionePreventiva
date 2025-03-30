clc;

%% Caricamento delle feature Koopman già estratte per il training
% Se non sono nel workspace, si caricano con: 
% load('koopman_features.mat'); % contiene X_features, y_targets

num_modes = 10;

%% Allenamento regressore su tutti i dati
disp('Allenamento regressore XGBoost su tutti i dati...');
model = fitrensemble(X_features, y_targets, ...
    'Method', 'LSBoost', 'NumLearningCycles', 100);

%% Predizione su tutti i dati di training (auto-valutazione)
disp('🔍 Predizione su tutti i dati di training...');
y_pred_all = predict(model, X_features);
y_true = y_targets;

%% Calcolo metriche
MAE  = mean(abs(y_true - y_pred_all));
MSE  = mean((y_true - y_pred_all).^2);
RMSE = sqrt(MSE);
R2   = 1 - sum((y_true - y_pred_all).^2) / sum((y_true - mean(y_true)).^2);

%% Risultati
fprintf('\nValutazione modello Koopman (training set):\n');
fprintf('MAE  = %.4f\n', MAE);
fprintf('RMSE = %.4f\n', RMSE);
fprintf('R^2  = %.4f\n', R2);

y_pred_smooth = movmean(y_pred_all, 15);  % Prova con 15 file di finestra

%% Visualizzazione predizione vs reale (con predizione smussata)
figure;
plot(y_true, 'b-', 'LineWidth', 1.5); hold on;
plot(y_pred_smooth, 'r--', 'LineWidth', 1.5);
legend('Reale', 'Predetto (smussato)');
xlabel('File #');
ylabel('Grado di guasto');
title('Predizione smussata del modello Koopman');
grid on;
