function [koopman_eigvals, koopman_modes] = koopman_EDMD(X, X_next, num_modes)
    % Funzione Koopman semplificata via osservabili estese, calcola autovalori e modi di Koopman tramite EDMD
    psi = @(x) [x; x.^2; sin(x)];  % Osservabili: originale, quadrato, seno

    N = min(size(X,2), size(X_next,2)); % Numero di snapshot da usare
    % Applica la funzione psi a ogni colonna (cioè a ogni stato nel tempo)
    Psi_X = zeros(size(psi(X(:,1)),1), N); % Osservabili al tempo t 
    Psi_X_next = zeros(size(psi(X_next(:,1)),1), N); % Osservabili al tempo t+1

    for i = 1:N
        Psi_X(:,i) = psi(X(:,i));
        Psi_X_next(:,i) = psi(X_next(:,i));
    end

    K = Psi_X_next * pinv(Psi_X);  % Operatore Koopman stimato. pinv è la pseudoinversa
    % Autovalori e autovettori di K
    [V, D] = eig(K);
    koopman_eigvals = diag(D);
    koopman_modes = V(:, 1:num_modes);
end

