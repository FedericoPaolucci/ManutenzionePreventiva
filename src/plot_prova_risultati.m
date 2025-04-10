% Ordina per valori reali (opzionale ma utile per allineare visualmente)
[~, sort_idx] = sort(y_val);
y_val_sorted = y_val(sort_idx);
y_val_pred_sorted = y_val_pred(sort_idx);

figure;
plot(y_val_sorted, 'b-', 'LineWidth', 1.5); hold on;
plot(y_val_pred_sorted, 'r--', 'LineWidth', 1.5);
legend('Reale', 'Predetto');
xlabel('Campione (ordinato per y\_val)');
ylabel('Health State');
title('Confronto Reale vs Predetto sul Validation Set');
grid on;

clc;

% Arrotonda predizioni (e limita ai valori tra 0 e 10)
y_val_pred_rounded = round(y_val_pred);
y_val_pred_rounded = max(0, min(10, y_val_pred_rounded));  % Clipping tra 0 e 10

% Idem per i valori veri (in caso ci siano arrotondamenti da fare)
y_val_rounded = round(y_val);
y_val_rounded = max(0, min(10, y_val_rounded));

% Categorie nominali da 0 a 10
classes = 0:10;

% Crea matrice di confusione
figure('Name','Matrice di Confusione - Validation Set');
cm = confusionchart(y_val_rounded, y_val_pred_rounded, ...
                    'RowSummary','row-normalized', ...
                    'ColumnSummary','column-normalized', ...
                    'Title','Matrice di Confusione (Validation)', ...
                    'XLabel','Predetto', ...
                    'YLabel','Reale');

% Imposta ordine delle classi (0-10)
cm.ClassLabels = string(classes);
