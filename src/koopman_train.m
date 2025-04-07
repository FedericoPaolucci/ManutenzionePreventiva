clc; 

%%  Parametri
disp('Impostazione parametri...');
window_size = 2048; % Window size scelta data la frequenza di campionamento del segnale: 20480 Hz.
num_modes = 10;  % Numero di modi Koopman da estrarre
num_files = length(data_all);  % Numero di file caricati

% Verifica se esiste un salvataggio parziale
if isfile('partial_koopman_features.mat')
    disp('Salvataggio parziale trovato. Riprendo da dove avevo interrotto...');
    load('partial_koopman_features.mat');  % Carica X_features, y_targets, k
    start_idx = k + 1;
else
    disp('Inizio da zero...');
    % Preallocazione delle feature (10 reali, 10 immaginari, 10 ampiezze, 10 growth rates)
    disp('Preallocazione delle matrici...');
    X_features = zeros(num_files, num_modes * 4);
    y_targets = zeros(num_files, 1);
    start_idx = 1;

    % Prepara blocco globale per stimare K
    disp('Costruzione blocco globale per Koopman...');
    X_all = []; X_all_next = [];
    for i = 1:num_files
        data = data_all{i};
        X1 = buffer(data(:,1), window_size);
        X1 = X1(:,1:end-1);
        X1_next = X1(:,2:end);
        X_all = [X_all, X1];
        X_all_next = [X_all_next, X1_next];
    end

    % Osservabili
    psi = @(x) [x; x.^2; sin(x)];
    % Applica osservabili a tutto il blocco
    N = size(X_all, 2);
    Psi_X = zeros(size(psi(X_all(:,1)),1), N);
    Psi_X_next = zeros(size(psi(X_all_next(:,1)),1), N);
    for i = 1:N
        Psi_X(:,i) = psi(X_all(:,i));
        Psi_X_next(:,i) = psi(X_all_next(:,i));
    end

    % Calcolo dell’operatore di Koopman globale
    K = Psi_X_next * pinv(Psi_X); % EDMD
    [V, D] = eig(K); % Decomposizione
    koopman_eigvals = diag(D); % Autovalori
    [~, idx_sort] = sort(abs(koopman_eigvals), 'descend');
    koopman_eigvals = koopman_eigvals(idx_sort);
    koopman_modes = V(:, idx_sort); % Modi ordinati per importanza
end

%% Barra di avanzamento
% per controllare a che punto siamo
h = waitbar(0, 'Inizio estrazione delle feature Koopman...');

%% Estrazione feature da ciascun file
disp('Inizio estrazione delle feature dai file...');
for k = start_idx:num_files
    waitbar(k/num_files, h, sprintf('File %d di %d', k, num_files));
    disp(['Elaborazione file ', num2str(k), '...']);

    data = data_all{k};
    label = labels(k);

    X1 = buffer(data(:,1), window_size);
    X1 = X1(:,1:end-1);

    % Costruisci Psi del file
    M = size(X1, 2);
    psi = @(x) [x; x.^2; sin(x)];
    Psi_file = zeros(size(psi(X1(:,1)),1), M);
    for i = 1:M
        Psi_file(:,i) = psi(X1(:,i));
    end

    % Proiezione Koopman
    koopman_projection = K * Psi_file;
    koopman_mean = mean(koopman_projection, 2);

    % % Growth rates via DMD
    [U, S, Vd] = svd(X1, 'econ');
    A_tilde = U' * X1(:,2:end) * Vd * diag(1 ./ diag(S));
    [~, eigVal] = eig(A_tilde);
    growth_rates = log(abs(diag(eigVal)));
    growth_rates = growth_rates(idx_sort(1:num_modes));

    % Feature Koopman
    koopman_real = real(koopman_eigvals(1:num_modes));
    koopman_imag = imag(koopman_eigvals(1:num_modes));
    koopman_amp  = abs(koopman_mean(1:num_modes));

    % Combina e salva
    X_features(k,:) = [koopman_real', koopman_imag', koopman_amp', growth_rates'];
    y_targets(k) = label;

    % Salvataggio ogni 20 file
    if mod(k, 20) == 0 || k == num_files
        save('partial_koopman_features.mat', 'X_features', 'y_targets', ...
             'K', 'koopman_eigvals', 'koopman_modes', 'k');
        disp(['Checkpoint salvato dopo ', num2str(k), ' file.']);
    end
end

close(h);  % Chiude la waitbar

% Normalizzazione finale
X_features = (X_features - mean(X_features)) ./ std(X_features);

% Salvataggio finale
save('koopman_features.mat', 'X_features', 'y_targets', 'K', 'koopman_eigvals', 'num_modes');
disp('"koopman_features.mat" salvato con successo.');

% Pulisci
delete('partial_koopman_features.mat');
disp('Pulizia completata.');