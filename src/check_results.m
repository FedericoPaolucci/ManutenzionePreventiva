clc; 

%% 📥 Caricamento del file CSV
filename = 'submission.csv';
data = readmatrix(filename);

sample_id = data(:,1);
probabilities = data(:,2:12); % Colonne 2-12: probabilità
confidence = data(:,13);     % Colonna 13

%% 📊 1. Plot delle distribuzioni di probabilità (primi 5 sample)
figure('Name','Distribuzioni di Probabilità (Primi 5 Sample)');
for i = 1:5
    subplot(2,3,i)
    bar(0:10, probabilities(i,:));
    ylim([0 1]);
    xlabel('Health State');
    ylabel('Probabilità');
    title(['Sample ', num2str(sample_id(i)), ' | Conf: ', num2str(confidence(i))]);
end
sgtitle('Distribuzioni di Probabilità dei Primi 5 Sample');

%% 📈 2. Entropia vs Max Probabilità
entropy_vals = -sum(probabilities .* log(probabilities + eps), 2); % Entropia
[max_probs, ~] = max(probabilities, [], 2);                        % Prob. max

figure('Name','Entropia vs Max Probabilità');
scatter(entropy_vals, max_probs, 35, confidence, 'filled');
xlabel('Entropia');
ylabel('Probabilità Massima');
title('Entropia vs Probabilità Massima');
colorbar;
grid on;

%% 📦 3. Istogramma degli Health State Predetti
[~, pred_state] = max(probabilities, [], 2); % Indice max = stato previsto (1=stato 0, ..., 11=stato 10)
health_states = pred_state - 1;              % Shift per partire da 0

figure('Name','Distribuzione Health State Predetti');
histogram(health_states, 0:10, 'FaceColor', [0.2 0.4 0.8]);
xlabel('Health State Predetto (0-10)');
ylabel('Frequenza');
title('Distribuzione degli Health State Predetti');
grid on;
