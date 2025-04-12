%% Caricamento file di TEST
clc; 

% Percorso della cartella di test
test_path = 'C:\datachallenge_dataset\Data_Challenge_PHM2023_test_data';

% Trova tutti i .txt
test_files = dir(fullfile(test_path, '*.txt'));

% Inizializza array per i segnali
data_test_all = {};

for i = 1:length(test_files)
    file_path = fullfile(test_path, test_files(i).name);
    disp(['File [', num2str(i), '/', num2str(length(test_files)), ']: ', test_files(i).name]);
    data = readmatrix(file_path);

    if size(data,2) == 4
        data = data(:,1:3); % Rimozione della quarta colonna se presente
    end

    data_test_all{end+1} = data;
end

disp(['File di test caricati: ', num2str(length(data_test_all))]);