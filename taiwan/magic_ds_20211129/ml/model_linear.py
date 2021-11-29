from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, SGDRegressor, Lasso, LassoCV, ElasticNet, ElasticNetCV

# набор параметров для GridSearchCV
params_Ridge = {
    'max_iter':[50000],
    'alpha':[0.1, 1, 10],
    'tol':[0.001, 0.01],
    'solver':['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
}
params_RidgeCV = {
    'alpha_per_target':[True, False]
}
params_SGDRegressor = {
    'max_iter':[50000],
    'alpha':[0.1, 1, 10],
    'tol':[0.001, 0.01],
    'penalty':['l2', 'l1', 'elasticnet'],
    'epsilon':[0.1, 1],
    'eta0':[0.01, 0.1]
}
params_Lasso = {
    'max_iter':[50000],
    'alpha':[0.1, 1, 10],
    'tol':[0.001, 0.01]
}
params_ElasticNet = {
    'max_iter':[50000],
    'alpha':[0.1, 1, 10],
    'l1_ratio':[0.1, 0.5, 1],
    'tol':[0.001, 0.01]
}
