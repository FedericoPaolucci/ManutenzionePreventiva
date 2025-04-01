clc
disp('Inizio estrazione feature Koopman per il set di test...');

%% Estrazione delle feature Koopman per il set di TEST
window_size = 2048;
num_modes = 10;
num_files_test = length(data_test_all);

%% Controllo salvataggio parziale
if isfile('partial_koopman_test_features.mat')
    disp('Salvataggio parziale trovato. Riprendo da dove avevo interrotto...');
    load('partial_koopman_test_features.mat');  % X_features_test, k
    start_idx_test = k + 1;
else
    disp('Inizio da zero...');
    % Preallocazione delle feature
    X_features_test = zeros(num_files_test, num_modes * 4);
    start_idx_test = 1;
end

h = waitbar(0, 'Estrazione feature Koopman test...'); % Waitbar

for k = start_idx_test:num_files_test
    data = data_test_all{k};

    % Uso dell'asse X
    X1 = buffer(data(:,1), window_size);
    X1 = X1(:, 1:end-1);
    X1_next = X1(:, 2:end);

    % Allineamento snapshot
    num_snapshots = min(size(X1,2), size(X1_next,2));
    X1 = X1(:,1:num_snapshots);
    X1_next = X1_next(:,1:num_snapshots);

    % Koopman
    [koopman_eigvals, koopman_modes] = koopman_EDMD(X1, X1_next, num_modes);

    koopman_real = real(koopman_eigvals);
    koopman_imag = imag(koopman_eigvals);
    koopman_amp  = abs(koopman_modes(:,1));

    % DMD tassi di crescita
    [U, S, V] = svd(X1, 'econ');
    A_tilde = U' * X1_next * V * diag(1 ./ diag(S));
    [~, eigVal] = eig(A_tilde);
    growth_rates = log(abs(diag(eigVal)));
    growth_rates = growth_rates(1:num_modes);

    % Combinazione risultati
    features = [koopman_real(1:num_modes)', ...
                koopman_imag(1:num_modes)', ...
                koopman_amp(1:num_modes)', ...
                growth_rates'];

    % Salvataggio
    X_features_test(k,:) = features;

    waitbar(k / num_files_test, h, ['Elaborazione file test ', num2str(k), ' di ', num2str(num_files_test)]);

    % Salvataggio ogni 20 file o ultimo
    if mod(k, 20) == 0 || k == num_files_test
        save('partial_koopman_test_features.mat', 'X_features_test', 'k');
        disp(['Salvato parzialmente dopo ', num2str(k), ' file.']);
    end

end

close(h);
disp('Estrazione delle feature Koopman per il test completata.');

% Salvataggio dei risultati
save('koopman_test_features.mat', 'X_features_test');
delete('partial_koopman_test_features.mat');
disp('"koopman_test_features.mat" salvato con successo');