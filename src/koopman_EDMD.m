function [koopman_eigvals, koopman_modes] = koopman_EDMD(X, X_next, num_modes)
    % Funzione Koopman semplificata via osservabili estese (EDMD)
    psi = @(x) [x; x.^2; sin(x)];  % Osservabili: lineari, quadratiche, sinusoidali

    N = min(size(X,2), size(X_next,2));
    Psi_X = zeros(size(psi(X(:,1)),1), N);
    Psi_X_next = zeros(size(psi(X_next(:,1)),1), N);

    for i = 1:N
        Psi_X(:,i) = psi(X(:,i));
        Psi_X_next(:,i) = psi(X_next(:,i));
    end

    K = Psi_X_next * pinv(Psi_X);  % Operatore Koopman stimato
    [V, D] = eig(K);
    koopman_eigvals = diag(D);
    koopman_modes = V(:, 1:num_modes);
end

