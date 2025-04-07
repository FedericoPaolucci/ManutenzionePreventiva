clc;

%% Caricamento delle feature Koopman già estratte per il training
load('koopman_features.mat'); % contiene 'X_features', 'y_targets', 'K', 'koopman_eigvals', 'num_modes'

%% Divisione training / validation stratificata 80/20 per ciascun livello di guasto
disp('Inizio divisione training / validation stratificata 80/20 per ciascun livello di guasto...');

unique_labels = unique(y_targets);
X_train = [];
y_train = [];
X_val = [];
y_val = [];

for i = 1:length(unique_labels)
    label = unique_labels(i);

    % Indici relativi alla classe
    idx = find(y_targets == label);
    idx = idx(randperm(length(idx))); % Shuffle

    % Split
    n_train = round(0.8 * length(idx));
    train_idx = idx(1:n_train);
    val_idx = idx(n_train+1:end);

    % Aggiunta ai set
    X_train = [X_train; X_features(train_idx,:)];
    y_train = [y_train; y_targets(train_idx)];
    X_val = [X_val; X_features(val_idx,:)];
    y_val = [y_val; y_targets(val_idx)];

    fprintf('Classe %d → %d train, %d val\n', label, length(train_idx), length(val_idx));
end

disp('Divisione stratificata completata.');

%% Allenamento regressore su tutti i dati
disp('Allenamento su training set...');
model = fitrensemble(X_train, y_train, 'Method', 'LSBoost', 'NumLearningCycles', 1000);

%% Predizione sul validation set
y_val_pred = predict(model, X_val);

mae = mean(abs(y_val - y_val_pred));
rmse = sqrt(mean((y_val - y_val_pred).^2));
r2 = 1 - sum((y_val - y_val_pred).^2)/sum((y_val - mean(y_val)).^2);

fprintf('MAE  = %.4f\n', mae);
fprintf('RMSE = %.4f\n', rmse);
fprintf('R²   = %.4f\n', r2);

%% Visualizzazione
figure;
plot(y_val, 'b-', 'LineWidth', 1.5); hold on;
plot(movmean(y_val_pred, 15), 'r--', 'LineWidth', 1.5);
legend('Reale', 'Predetto');
xlabel('Campione');
ylabel('Health State');
title('Validazione Koopman + Regressore');
grid on;