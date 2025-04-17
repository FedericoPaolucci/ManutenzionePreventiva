clc;
load('koopman_features.mat'); % contiene X_features, y_targets
load('koopman_test_features.mat') % feature del test

%% Oversampling manuale delle classi (moltiplichiamo i sample)
idx_0 = find(y_targets == 0);
idx_1 = find(y_targets == 1);
idx_2 = find(y_targets == 2);
idx_3 = find(y_targets == 3);
idx_4 = find(y_targets == 4);
idx_6 = find(y_targets == 6);
idx_8 = find(y_targets == 8);

% Funzione helper: oversampling con duplicazione
augment = @(X, y, n) deal(repmat(X, n, 1), repmat(y, n, 1));
% Funzione helper: oversampling con rumore (non usata)
augment_noise = @(X, y, n, noise) deal(repmat(X, n, 1) + noise * randn(size(repmat(X, n, 1))), repmat(y, n, 1));

% Oversampling semplice
[X_aug0, y_aug0] = augment(X_features(idx_0,:), y_targets(idx_0), 10);
[X_aug1, y_aug1] = augment(X_features(idx_1,:), y_targets(idx_1), 8);
[X_aug2, y_aug2] = augment(X_features(idx_2,:), y_targets(idx_2), 5);
[X_aug3, y_aug3] = augment(X_features(idx_3,:), y_targets(idx_3), 5);
[X_aug4, y_aug4] = augment(X_features(idx_4,:), y_targets(idx_4), 5);
[X_aug6, y_aug6] = augment(X_features(idx_6,:), y_targets(idx_6), 8);
[X_aug8, y_aug8] = augment(X_features(idx_8,:), y_targets(idx_8), 10);

%% Crea dataset esteso
X_augmented = [X_features; X_aug0; X_aug1; X_aug2; X_aug3; X_aug4; X_aug6; X_aug8];
y_augmented = [y_targets; y_aug0; y_aug1; y_aug2; y_aug3; y_aug4; y_aug6; y_aug8];

%% Normalizza le feature
% Calcola media e deviazione standard dal dataset di training
mu = mean(X_augmented);
sigma = std(X_augmented);

% Evita divisioni per zero
sigma(sigma == 0) = eps;

% Applica la normalizzazione standard (z-score)
X_augmented = (X_augmented - mu) ./ sigma;
X_features_test = (X_features_test - mu) ./ sigma;

%% Divisione 80-20 stratificata
% Mantiene la stessa distribuzione delle classi tra train e validation
cv = cvpartition(y_augmented, 'HoldOut', 0.2, 'Stratify', true);
X_train = X_augmented(training(cv), :);
y_train = y_augmented(training(cv));
X_val   = X_augmented(test(cv), :);
y_val   = y_augmented(test(cv));

%% Addestramento modello fitrnet -- [precedentemente fitrensemble (regressione con alberi e Bagging)]
%disp("Addestramento fitrensemble...");
%model_ensemble = fitrensemble(X_train, y_train, 'Method', 'Bag', 'NumLearningCycles', 200); % Bag; 200 alberi
disp("Addestramento fitrnet...");
model_net = fitrnet(X_train, y_train, ...
    'LayerSizes', [64, 32, 16], ...
    'Activations', 'relu', ...
    'Standardize', true, ...
    'Verbose', 1);

%% Valutazione sul validation set
y_val_pred_ensemble = predict(model_net, X_val);

% Valori di errore MAE e RMSE
fprintf("Errore fitrnet: MAE = %.4f | RMSE = %.4f\n", ...
    mean(abs(y_val - y_val_pred_ensemble)), sqrt(mean((y_val - y_val_pred_ensemble).^2)));

%% Plot fitrnet
[~, sort_idx] = sort(y_val);
y_val_sorted = y_val(sort_idx);
y_val_pred_ensemble_sorted = y_val_pred_ensemble(sort_idx);

figure('Name', 'Reale vs Predetto - fitrnet');
plot(y_val_sorted, 'k-', 'LineWidth', 1.5); hold on;
plot(y_val_pred_ensemble_sorted, 'b--', 'LineWidth', 1.5);
legend('Reale', 'Predetto (fitrnet)');
xlabel('Campione (ordinato per y\_val)');
ylabel('Health State');
title('Confronto Reale vs Predetto - fitrnet');
grid on;

%% Matrici di Confusione

y_val_rounded = max(0, min(10, round(y_val)));
y_pred_ens_rounded = max(0, min(10, round(y_val_pred_ensemble)));
classes = 0:10;

figure('Name','Confusione - fitrnet');
cm_ens = confusionchart(y_val, y_pred_ens_rounded, ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized', ...
    'Title','Matrice di Confusione (Validation) - fitrnet', ...
    'XLabel','Predetto', 'YLabel','Reale');

%% Predizione su tutto il set X_features e salvataggio CSV
X_features_prova = normalize(X_features);
y_pred = predict(model_net, X_features_prova);

val_errors = y_val - y_val_pred_ensemble;
b_err = std(val_errors) / sqrt(2);  % Parametro di scala Laplace

x_vals = 0:10;
num_samples = size(X_features_prova, 1);
output_matrix = zeros(num_samples, 14);  % [ID, Prob_0 ... Prob_10, Confidence, GroundTruth]

for i = 1:num_samples
    probs = exp(-abs(x_vals - y_pred(i)) / b_err);  % Distribuzione Laplace
    probs = probs / sum(probs);  % Normalizzazione

    entropy_val = -sum(probs .* log(probs + eps));
    confidence = double(entropy_val < 0.6);

    output_matrix(i,:) = [i, probs, confidence, y_targets(i)];
end

header = ["SampleID", "HealthState_0", "HealthState_1", "HealthState_2", ...
          "HealthState_3", "HealthState_4", "HealthState_5", "HealthState_6", ...
          "HealthState_7", "HealthState_8", "HealthState_9", "HealthState_10", ...
          "Confidence", "GroundTruth"];

filename = 'submission_with_gt.csv';
fid = fopen(filename, 'w');
fprintf(fid, '%s,', header{1:end-1});
fprintf(fid, '%s\n', header{end});
fclose(fid);

writematrix([header; num2cell(output_matrix)], filename);
disp(['File generato: ', filename]);


%% Matrice di confusione per predizione su set di partenza

y_gt_rounded = max(0, min(10, round(y_targets)));
y_pred_rounded = max(0, min(10, round(y_pred)));

figure('Name','Matrice di Confusione - Set completo');
cm_ens = confusionchart(y_gt_rounded, y_pred_rounded, ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized', ...
    'Title','Matrice di Confusione finale - fitrnet', ...
    'XLabel','Predetto', 'YLabel','Reale');

%% Set di test per submission
y_pred_test = predict(model_net, X_features_test);

% Calcolo probabilità e creazione del CSV per submission
x_vals = 0:10;
num_samples_test = size(X_features_test, 1);
output_matrix_test = zeros(num_samples_test, 13);  % [ID, Prob_0 ... Prob_10, Confidence]

% Calcola la scala della distribuzione di Laplace come prima
val_errors = y_val - y_val_pred_ensemble;
b_err = std(val_errors) / sqrt(2);

for i = 1:num_samples_test
    probs = exp(-abs(x_vals - y_pred_test(i)) / b_err);  % Distribuzione di Laplace
    probs = probs / sum(probs);  % Normalizzazione

    entropy_val = -sum(probs .* log(probs + eps));
    confidence = double(entropy_val < 0.6);

    output_matrix_test(i,:) = [i, probs, confidence];
end

% Intestazione CSV
header_test = ["sample_id", "prob_0", "prob_1", "prob_2", ...
               "prob_3", "prob_4", "prob_5", "prob_6", ...
               "prob_7", "prob_8", "prob_9", "prob_10", ...
               "confidence"];

% Scrittura CSV
filename_test = 'submission.csv';
fid = fopen(filename_test, 'w');
fprintf(fid, '%s,', header_test{1:end-1});
fprintf(fid, '%s\n', header_test{end});
fclose(fid);

writematrix([header_test; num2cell(output_matrix_test)], filename_test);
disp(['File CSV del test generato: ', filename_test]);

% Istogramma della distribuzione delle classi predette (approssimate)
y_pred_rounded_test = max(0, min(10, round(y_pred_test)));

figure('Name','Distribuzione delle predizioni - Test set');
histogram(y_pred_rounded_test, 'BinEdges', -0.5:1:10.5, 'FaceColor', [0.2 0.4 0.8]);
xlabel('Classe Predetta');
ylabel('Conteggio');
title('Distribuzione delle Classi Predette - Set di Test');
grid on;