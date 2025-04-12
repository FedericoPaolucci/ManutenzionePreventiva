%% Caricamento di tutti i file da tutte le cartelle
clc; 

% Percorso della cartella principale
dataset_path = 'C:\datachallenge_dataset\Data_Challenge_PHM2023_training_data';

%️ Nomi delle cartelle (ogni cartella rappresenta un livello di guasto)
folders = {'Pitting_degradation_level_0 (Healthy)', 'Pitting_degradation_level_1','Pitting_degradation_level_2', 'Pitting_degradation_level_3', 'Pitting_degradation_level_4', 'Pitting_degradation_level_6', 'Pitting_degradation_level_8'};

% Inizializza array
data_all = {};
labels = [];

% Cicla ogni cartella
for i = 1:length(folders)
    folder_name = folders{i};
    folder_path = fullfile(dataset_path, folder_name);
    files = dir(fullfile(folder_path, '*.txt')); % Prende tutti i .txt
    disp(['Caricamento cartella: ', folder_name, ' (', num2str(length(files)), ' file trovati)']);
    
    for j = 1:length(files)
        file_path = fullfile(folder_path, files(j).name);
        disp(['File [', num2str(j), '/', num2str(length(files)), ']: ', files(j).name]);
        data = readmatrix(file_path);

        % Se ci sono 4 colonne, rimuove l’ultima
        if size(data,2) == 4
            data = data(:,1:3);
        end

        % Salva il file nella lista
        data_all{end+1} = data;

        % Estrai il numero del livello di guasto dal nome della cartella
        level = regexp(folder_name, '\d+', 'match');
        level_num = str2double(level{end}); % prende il livello di guasto scritto sulla cartella
        labels(end+1) = level_num;
    end
end

disp(['Totale file caricati: ', num2str(length(data_all))]);