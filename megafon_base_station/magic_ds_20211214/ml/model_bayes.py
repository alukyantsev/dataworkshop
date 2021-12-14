import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB

# набор параметров для GridSearchCV
params_GaussianNB = {
    'var_smoothing':np.logspace(0, -9, num=100)
}
params_BernoulliNB = {
    'alpha':np.logspace(-2, 5, num=100)
}
params_MultinomialNB = {
    'alpha':np.logspace(-2, 5, num=100)
}
params_ComplementNB = {
    'alpha':np.logspace(-2, 5, num=100)
}
