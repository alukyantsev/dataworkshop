import xgboost as xgb
import catboost as ctb
import lightgbm as lgb

# набор параметров для одной модели
param_XGBClassifier = {
    'max_depth': 5, 'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42,
    'objective': 'binary:logistic', 'eval_metric': 'mlogloss', 'use_label_encoder': False
}
param_XGBClassifier_multiclass = {
    'max_depth': 5, 'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42,
    'eval_metric': 'mlogloss', 'use_label_encoder': False
}
param_XGBRegressor = {
    'max_depth': 5, 'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42
}
param_CatBoostClassifier = {
    'max_depth': 5, 'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42
}
param_CatBoostRegressor = {
    'max_depth': 5, 'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42
}
param_LGBMClassifier = {
    'max_depth': 5, 'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42
}
param_LGBMRegressor = {
    'max_depth': 5, 'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42
}

# набор параметров для GridSearchCV
params_XGBClassifier = {
    'max_depth':[3, 5, 6, 7, 9],
    'n_estimators':[50, 100, 150, 300, 500],
    'learning_rate':[0.01, 0.1],
    'gamma':[0, 1, 10],
    'scale_pos_weight':[1, 5, 10],
    'max_delta_step':[0, 1],
    'min_child_weight':[1, 2, 5],
    'random_state': [42],
    'objective': ['binary:logistic'],
    'eval_metric': ['mlogloss'],
    'use_label_encoder': [False]
}
params_base_XGBClassifier = {
    'max_depth':[3, 5],
    'n_estimators':[100, 300, 500],
    'random_state': [42],
    'objective': ['binary:logistic'],
    'eval_metric': ['mlogloss'],
    'use_label_encoder': [False]
}

params_XGBClassifier_multiclass = {
    'max_depth':[3, 5, 6, 7, 9],
    'n_estimators':[50, 100, 150, 300, 500],
    'learning_rate':[0.01, 0.1],
    'gamma':[0, 1, 10],
    'scale_pos_weight':[1, 5, 10],
    'max_delta_step':[0, 1],
    'min_child_weight':[1, 2, 5],
    'random_state': [42],
    'eval_metric': ['mlogloss'],
    'use_label_encoder': False
}
params_base_XGBClassifier_multiclass = {
    'max_depth':[3, 5],
    'n_estimators':[100, 300, 500],
    'random_state': [42],
    'eval_metric': ['mlogloss'],
    'use_label_encoder': False
}

params_XGBRegressor = {
    'max_depth':[3, 5, 6, 7, 9],
    'n_estimators':[50, 100, 150, 300, 500],
    'learning_rate':[0.01, 0.1],
    'gamma':[0, 1, 10],
    'scale_pos_weight':[1, 5, 10],
    'max_delta_step':[0, 1],
    'min_child_weight':[1, 2, 5],
    'random_state': [42]
}
params_base_XGBRegressor = {
    'max_depth':[3, 5],
    'n_estimators':[100, 300, 500],
    'random_state': [42]
}

params_CatBoostClassifier = {
    'max_depth':[3, 5, 7],
    'n_estimators':[100, 300, 500],
    'learning_rate':[0.01, 0.1],
    'random_state': [42],
    'verbose': [0]
}
params_base_CatBoostClassifier = {
    'max_depth':[5, 7],
    'n_estimators':[100, 300, 500],
    'learning_rate':[0.1],
    'random_state': [42],
    'verbose': [0]
}
params_CatBoostRegressor = {
    'max_depth':[3, 5, 7],
    'n_estimators':[100, 300, 500],
    'learning_rate':[0.01, 0.1],
    'random_state': [42],
    'verbose': [0]
}
params_base_CatBoostRegressor = {
    'max_depth':[5, 7],
    'n_estimators':[100, 300, 500],
    'learning_rate':[0.1],
    'random_state': [42],
    'verbose': [0]
}

params_LGBMClassifier = {
    'max_depth':[3, 5, 7],
    'n_estimators':[100, 300, 500],
    'min_child_weight':[1, 2, 5],
    'learning_rate':[0.01, 0.1],
    'random_state': [42]
}
params_base_LGBMClassifier = {
    'max_depth':[3, 5],
    'n_estimators':[100, 300, 500],
    'random_state': [42]
}
params_LGBMRegressor = {
    'max_depth':[3, 5, 7],
    'n_estimators':[100, 300, 500],
    'min_child_weight':[1, 2, 5],
    'learning_rate':[0.01, 0.1],
    'random_state': [42]
}
params_base_LGBMRegressor = {
    'max_depth':[3, 5],
    'n_estimators':[100, 300, 500],
    'random_state': [42]
}
