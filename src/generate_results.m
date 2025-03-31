clc;

num_states = 11;
x_vals = linspace(0, 1, num_states); % Livelli salute normalizzati

% Deviazione standard dell'errore ottenuta su validation
if ~exist('sigma_err', 'var')
    error('sigma_err non definito. Calcolalo dal validation set!');
end

%% Predizione per ogni sample di test
num_samples = size(X_features_test, 1);
output_matrix = zeros(num_samples, 13);  % 13 colonne richieste

for i = 1:num_samples
    % Predizione del valore continuo [0, 1]
    y_pred = predict(model, X_features_test(i, :));

    % Costruzione distribuzione di probabilità gaussiana basata sull'errore reale
    prob_dist = exp(-0.5 * ((x_vals - y_pred) / sigma_err).^2);
    prob_dist = prob_dist / sum(prob_dist);  % Normalizzazione
    % Calcolo entropia per stimare la confidenza
    entropy_val = -sum(prob_dist .* log(prob_dist + eps));  % eps evita log(0)
    confidence = double(entropy_val < 0.75);  % più bassa l’entropia → più confidenza (scelto 0.75)

    % Riga finale: SampleID (i), 11 prob., confidenza
    output_matrix(i, :) = [i, prob_dist, confidence];
end

%% Salvataggio della matrice in un file .csv
filename = 'submission.csv';

% Intestazione (colonne)
header = "SampleID";
for k = 0:10
    header = [header, "HealthState_" + k];
end
header = [header, "Confidence"];

% Scrittura header
fid = fopen(filename, 'w');
fprintf(fid, '%s,', header{1:end-1});
fprintf(fid, '%s\n', header{end});
fclose(fid);

% Appendi i dati
dlmwrite(filename, output_matrix, '-append');
disp(['File di submission generato: ', filename]);