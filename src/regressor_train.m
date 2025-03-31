clc;

%% Caricamento delle feature Koopman già estratte per il training
% Se non sono nel workspace, si caricano con: 
% load('koopman_features.mat'); % contiene X_features, y_targets

num_modes = 10;

%% Divisione in Training/Validation set
cv = cvpartition(size(X_features, 1), 'HoldOut', 0.2);
idxTrain = training(cv);
idxVal = test(cv);

X_train = X_features(idxTrain, :);
y_train = y_targets(idxTrain);

X_val = X_features(idxVal, :);
y_val = y_targets(idxVal);

%% Allenamento regressore su tutti i dati
disp('Allenamento su training set...');
model = fitrensemble(X_train, y_train, 'Method', 'LSBoost', 'NumLearningCycles', 100);

%% Predizione sul validation set
y_val_pred = predict(model, X_val);

%% Calcolo degli errori sulla validation
val_errors = y_val - y_val_pred;

% Calcolo statistico dell’errore
mu_err = mean(val_errors);           % media dell’errore
sigma_err = std(val_errors);        % deviazione standard
fprintf('\n Errore su validation set:\n');
fprintf('Errore medio     = %.4f\n', mu_err);
fprintf('Deviazione std.  = %.4f\n', sigma_err);
% Distribuzione usata successivamente per stimare probabilità/confidenza

%% Visualizzazione
y_pred_smooth = movmean(y_pred_all, 15);

figure;
plot(y_true, 'b-', 'LineWidth', 1.5); hold on;
plot(y_pred_smooth, 'r--', 'LineWidth', 1.5);
legend('Reale', 'Predetto (smussato)');
xlabel('File #');
ylabel('Grado di guasto');
title('Predizione smussata del modello Koopman');
grid on;