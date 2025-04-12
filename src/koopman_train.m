clc; 

%%  Parametri
disp('Impostazione parametri...');
window_size = 2048; % Window size scelta data la frequenza di campionamento del segnale: 20480 Hz.
num_modes = 10;  % Numero di modi Koopman da estrarre
num_files = length(data_all);  % Numero di file caricati

%% Controllo salvataggio parziale
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
end

tic;  % Per monitorare il tempo totale

h = waitbar(0, 'Inizio estrazione delle feature Koopman...'); % waitbar per controllare a che punto siamo

%% Estrazione feature da ciascun file
disp('Inizio estrazione delle feature dai file...');
for k = start_idx:num_files
    waitbar(k/num_files, h, ['Elaborazione file ', num2str(k), ' di ', num2str(num_files)]); % waitbar

    disp(['File #', num2str(k), ' su ', num2str(num_files)]);
    data = data_all{k}; % carica il segnale
    label = labels(k); % etichetta associata (dalla cartella)
    disp('Dati caricati.');

    % Segmentazione del segnale sull'asse X in finestre sovrapposte.
    disp('Segmentazione del segnale con buffer...');
    X1 = buffer(data(:,1), window_size);
    X1 = X1(:, 1:end-1);
    X1_next = X1(:, 2:end);

    % Allinea X e X_next in modo che abbiano lo stesso numero di colonne
    num_snapshots = min(size(X1,2), size(X1_next,2));
    X1 = X1(:,1:num_snapshots);
    X1_next = X1_next(:,1:num_snapshots);
    disp(['Snapshot allineati: ', num2str(num_snapshots)]);

    % Calcolo dei modi e degli autovalori di Koopman tramite EDMD
    disp('Calcolo Koopman (EDMD)...');
    [koopman_eigvals, koopman_modes] = koopman_EDMD(X1, X1_next, num_modes);

    % Feature Koopman
    koopman_real = real(koopman_eigvals); % parte reale autovalori
    koopman_imag = imag(koopman_eigvals); % parte immaginaria
    koopman_amp  = abs(koopman_modes(:,1)); % ampiezza del primo modo
    disp('Feature Koopman ottenute.');

    % Growth rates (tassi di crescita) da DMD
    disp('Calcolo DMD e growth rates...');
    [U, S, V] = svd(X1, 'econ');
    A_tilde = U' * X1_next * V * diag(1 ./ diag(S));
    [~, eigVal] = eig(A_tilde);
    growth_rates = log(abs(diag(eigVal)));
    growth_rates = growth_rates(1:num_modes);  % Mantieni stessi modi
    disp('Growth rates calcolati.');

    % Combinazione di tutte le feature
    features = [koopman_real(1:num_modes)', ...
                koopman_imag(1:num_modes)', ...
                koopman_amp(1:num_modes)', ...
                growth_rates'];

    % Salvataggio
    X_features(k, :) = features;
    y_targets(k) = label;

    % Salvataggio parziale ogni 20 file
    if mod(k, 20) == 0 || k == num_files
        save('partial_koopman_features.mat', 'X_features', 'y_targets', 'k');
        disp(['Salvato parzialmente dopo ', num2str(k), ' file.']);
    end

end

close(h);  % Chiude la waitbar
% salvataggio finale e pulizia
disp(['Estrazione completata in ', num2str(toc, '%.2f'), ' secondi']);
save('koopman_features.mat', 'X_features', 'y_targets');
disp('"koopman_features.mat" salvato con successo.');
delete('partial_koopman_features.mat');
disp('Pulizia completata.');