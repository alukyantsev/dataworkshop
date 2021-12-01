from sklearn.ensemble import RandomForestClassifier

# набор параметров для одной модели
param_RandomForestClassifier = {
    'max_depth': 5, 'n_estimators': 100, 'random_state': 42
}

# набор параметров для GridSearchCV
params_RandomForestClassifier = {
    'n_estimators': [100, 200, 500, 1000],
    'criterion': ['gini', 'entropy'],
    'max_depth': [1, 5, 10, 15, 20],
    'max_features': ['log2', 'sqrt'],
    'min_samples_leaf': [2, 4, 6, 8],
    'min_samples_split': [2, 5, 7, 9, 11, 14],
    'class_weight': ['balanced', 'balanced_subsample'],
    'random_state': [42]
}
params_base_RandomForestClassifier = {
    'n_estimators': [100, 300, 500],
    'max_depth': [1, 5],
    'random_state': [42]
}
