clc;

%% Caricamento delle feature Koopman già estratte per il training
load('koopman_features.mat'); % contiene X_features, y_targets

num_modes = 10;

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
    X_train = [X_train; X_features(train_idx, :)];
    y_train = [y_train; y_targets(train_idx)];

    X_val = [X_val; X_features(val_idx, :)];
    y_val = [y_val; y_targets(val_idx)];

    fprintf('Classe %d → %d train, %d val\n', label, length(train_idx), length(val_idx));
end

disp('Divisione stratificata completata.');

%% Allenamento regressore su tutti i dati
disp('Allenamento su training set...');
model = fitrensemble(X_train, y_train, 'Method', 'LSBoost', 'NumLearningCycles', 1000);

%% Predizione sul validation set
y_val_pred = predict(model, X_val);

%% Calcolo degli errori sulla validation
val_errors = y_val - y_val_pred;

% Statistiche errore
mu_err = mean(val_errors);
sigma_err = std(val_errors);
fprintf('\nErrore su validation set:\n');
fprintf('Errore medio     = %.4f\n', mu_err);
fprintf('Deviazione std.  = %.4f\n', sigma_err);

%% Visualizzazione
y_pred_smooth = movmean(y_val_pred, 15);

figure;
plot(y_val, 'b-', 'LineWidth', 1.5); hold on;
plot(y_pred_smooth, 'r--', 'LineWidth', 1.5);
legend('Reale (Validation)', 'Predetto (smussato)');
xlabel('Campione');
ylabel('Grado di guasto');
title('Predizione sul Validation Set - Modello Koopman');
grid on;