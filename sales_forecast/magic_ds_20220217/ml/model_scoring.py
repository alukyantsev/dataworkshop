import numpy as np
from sklearn.metrics import make_scorer, get_scorer, accuracy_score, balanced_accuracy_score, f1_score, recall_score, precision_score, average_precision_score, roc_auc_score, max_error, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score, classification_report

# добавляем кастомную метрику SMAPE
# https://ru.abcdef.wiki/wiki/Symmetric_mean_absolute_percentage_error
def smape_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / np.abs(y_true + y_pred + 1e-15))
smape = make_scorer(smape_error, greater_is_better=False)
