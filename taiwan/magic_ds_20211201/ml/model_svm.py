from sklearn.svm import LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM, SVC, SVR, l1_min_c

# набор параметров для GridSearchCV
params_LinearSVC = {
    'max_iter': [500000],
    'C': [0.1, 1, 3],
    'loss': ['hinge']
}
params_SVC = {
    'max_iter': [500000],
    'C': [0.1, 1, 3],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 5, 8, 12],
    'gamma': [0.001, 0.01],
    'coef0': [1]
}
params_NuSVC = {
    'nu': [0.05, 0.15, 0.25],
    'max_iter': [200000],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 5, 8, 12],
    'gamma': [0.001, 0.0001],
    'coef0': [1]
}
params_LinearSVR = {
    'max_iter': [200000],
    'C': [0.1, 1, 3, 10, 100],
    'epsilon': [0.1, 1, 1.5]
}
params_SVR = {
    'max_iter': [500000],
    'C': [0.1, 1, 3],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 5, 8, 12],
    'gamma': [0.001, 0.1],
    'epsilon': [0.1, 1]
}
params_NuSVR = {
    'nu': [0.05, 0.15, 0.25],
    'max_iter': [200000],
    'C': [0.1, 1, 3],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 5, 8, 12],
    'gamma': [0.001, 0.1]
}
params_OneClassSVM = {
    'nu': [0.05, 0.15, 0.25],
    'max_iter': [200000],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'degree': [2, 3, 5, 8, 12],
    'gamma': [0.001, 5]
}
