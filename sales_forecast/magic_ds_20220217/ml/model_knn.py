from sklearn.neighbors import KNeighborsClassifier

# набор параметров для GridSearchCV
params_KNeighborsClassifier = {
    'n_neighbors':range(3, 30, 1),
    'metric':['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
    'weights':['uniform', 'distance'],
    'algorithm':['ball_tree', 'kd_tree', 'brute']
}
