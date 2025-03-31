clc; 

%%  Parametri
window_size = 4096; % La frequenza di campionamento del segnale è 20480 Hz. Una finestra di 4096 campioni corrisponde a 0.2 secondi.
num_modes = 10;  % Numero di modi Koopman da estrarre
num_files = length(data_all);  % Numero di file caricati

% Preallocazione delle feature (10 reali, 10 immaginari, 10 ampiezze, 10 growth rates)
X_features = zeros(num_files, num_modes * 4);
y_targets = zeros(num_files, 1);

tic;  % Per monitorare il tempo totale

%% Estrazione feature da ciascun file
for k = 1:num_files
    data = data_all{k};
    label = labels(k);

    % Usa solo una direzione (es. asse X)
    X1 = buffer(data(:,1), window_size);
    X1 = X1(:, 1:end-1);
    X1_next = X1(:, 2:end);

    % Allinea numero di snapshot
    num_snapshots = min(size(X1,2), size(X1_next,2));
    X1 = X1(:,1:num_snapshots);
    X1_next = X1_next(:,1:num_snapshots);

    % Koopman semplificato (EDMD)
    [koopman_eigvals, koopman_modes] = koopman_EDMD(X1, X1_next, num_modes);

    % Feature Koopman
    koopman_real = real(koopman_eigvals);
    koopman_imag = imag(koopman_eigvals);
    koopman_amp  = abs(koopman_modes(:,1));

    % Growth rates (tassi di crescita) da DMD
    [U, S, V] = svd(X1, 'econ');
    A_tilde = U' * X1_next * V * diag(1 ./ diag(S));
    [~, eigVal] = eig(A_tilde);
    growth_rates = log(abs(diag(eigVal)));
    growth_rates = growth_rates(1:num_modes);  % Mantieni stessi modi

    % Combinazione di tutte le feature
    features = [koopman_real(1:num_modes)', ...
                koopman_imag(1:num_modes)', ...
                koopman_amp(1:num_modes)', ...
                growth_rates'];

    % Salvataggio
    X_features(k, :) = features;
    y_targets(k) = label;

    % Log ogni 100 file
    if mod(k, 100) == 0
        disp(['File elaborati: ', num2str(k), ' su ', num2str(num_files), ...
              ' (tempo: ', num2str(toc, '%.2f'), ' s)']);
    end
end

disp(['Estrazione completata in ', num2str(toc, '%.2f'), ' secondi']);

