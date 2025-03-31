%% Estrazione delle feature Koopman per il set di TEST
window_size = 4096;
num_modes = 10;
num_files_test = length(data_test_all);

% Preallocazione delle feature
X_features_test = zeros(num_files_test, num_modes * 4);

for k = 1:num_files_test
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

    if mod(k, 100) == 0
        disp(['File test elaborati: ', num2str(k), ' su ', num2str(num_files_test)]);
    end
end

disp('Estrazione delle feature Koopman per il test completata.');
