% Percorso del file di testo (sostituire con il percorso effettivo)
filePath = '/Users/simonegiano/Documents/MATLAB/Manutenzione preventiva/Data_Challenge_PHM2023_training_data/Pitting_degradation_level_0 (Healthy)/V100_50N_1.txt';

% Carica i dati dal file di testo in una matrice 'data'
% La funzione "load" legge il contenuto del file come una matrice.
% Assicurati che il file sia organizzato in righe e colonne con valori numerici.
% Carica i dati dal file di testo in una matrice 'data'
data = load(filePath);

% Crea un vettore tempo/indice (se non hai un vettore tempo specifico)
t = 1:size(data, 1);

% Crea una figura
figure;

% Primo subplot (colonna 1)
subplot(3,1,1);
plot(t, data(:,1));
title('Colonna 1');
xlabel('Campioni (n)');
ylabel('Ampiezza');
grid on;

% Secondo subplot (colonna 2)
subplot(3,1,2);
plot(t, data(:,2));
title('Colonna 2');
xlabel('Campioni (n)');
ylabel('Ampiezza');
grid on;

% Terzo subplot (colonna 3)
subplot(3,1,3);
plot(t, data(:,3));
title('Colonna 3');
xlabel('Campioni (n)');
ylabel('Ampiezza');
grid on;
