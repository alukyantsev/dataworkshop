import pandas as pd
import numpy as np
import math
from sklearn.model_selection import GridSearchCV
import scikitplot as skplt
from . visualize import *
from . model_scoring import *
from . lazypredict import LazyClassifier, LazyRegressor

# делаем грубую оценку моделей по датасету через LazyPredict
def model_score(
    estimator, X_train, y_train, X_valid, y_valid, verbose=0, 
    classifiers='all', custom_metric=None, random_state=42
):

    clf = estimator(verbose=verbose, ignore_warnings=True, custom_metric=custom_metric, predictions=True,
                    random_state=random_state, classifiers=classifiers)
    models, predictions = clf.fit(X_train, X_valid, y_train, y_valid)
    return models, predictions

# подбор параметров модели через GridSearchCV
#
# model_dict = model_fit(estimator, param_grid,
#                        X_train, y_train, X_valid, y_valid, X_test, y_test,
#                        scoring='accuracy', result=10, mute=False)
# если переданы параметры X_test, y_test, то считается все 3 стадии проверки
# возвращается модель и предикт по X_test, X_valid
#
# model_dict = model_fit(estimator, param_grid,
#                        X_train, y_train, X_valid, y_valid, X_test,
#                        scoring='accuracy', result=10, mute=False)
# если не передан y_test, то считаются 2 стадии, а на 3 стадии делается предикт по X_test
# возвращается модель и предикт по X_test, X_valid
#
# model_dict = model_fit(estimator, param_grid,
#                        X_train, y_train, X_valid, y_valid,
#                        scoring='accuracy', result=10, mute=False)
# если не передан X_test, y_test, то считаются 2 стадии
# возвращается модель и предикт по X_valid
#
# параметр result обозначает сколько моделей с первого шага взять для проверки на второй
# для несбалансированных датасетов лучше брать больше моделей
# 
# параметр learning_curves_dots обозначает количество точек для кривой обучения
# 
# model_dict = model_fit(estimator, param_grid,
#                        X_train, y_train, X_valid, y_valid, X_test,
#                        result=10, scoring='f1', learning_curves_dots=10,
#                        threshold=np.arange(0.1, 0.7, step=0.01))
# параметр threshold применяется для задач бинарной классификации для несбалансированного датасете
# принимает массив значений, на основе которых проверяется максимальный скоринг при определенной
# вероятности принадлежности к классу у модели
# 
def model_fit(estimator, param_grid,
                X_train, y_train,
                X_valid, y_valid,
                X_test=pd.DataFrame(), y_test=pd.Series(dtype='int64'),
                scoring='accuracy',
                result=20,
                mute=False,
                n_jobs=4,
                cv=5,
                learning_curves_dots=10,
                threshold=[]
):

    # задаем словарь скоринга
    if scoring == 'accuracy':
        scoring_f = accuracy_score
        scoring_greater_is_better = True
    if scoring == 'f1':
        scoring_f = f1_score
        scoring_greater_is_better = True
    if scoring == 'precision':
        scoring_f = precision_score
        scoring_greater_is_better = True
    if scoring == 'recall':
        scoring_f = recall_score
        scoring_greater_is_better = True
    if scoring == 'roc_auc':
        scoring_f = roc_auc_score
        scoring_greater_is_better = True
    if scoring == 'max_error':
        scoring_f = max_error
        scoring_greater_is_better = False
    if scoring == 'neg_mean_absolute_error':
        scoring_f = mean_absolute_error
        scoring_greater_is_better = False
    if scoring == 'neg_mean_squared_error':
        scoring_f = mean_squared_error
        scoring_greater_is_better = False
    if scoring == 'neg_median_absolute_error':
        scoring_f = median_absolute_error
        scoring_greater_is_better = False
    if scoring == 'r2':
        scoring_f = r2_score
        scoring_greater_is_better = True

    ###
    ### STAGE 1: обучаем модель на тренировочных данных
    ###
    
    # обучаем модель
    gsearch = GridSearchCV(
        estimator = estimator(),
        param_grid = param_grid,
        scoring=scoring,
        n_jobs=n_jobs,
        cv=cv
    )
    gsearch.fit(X_train.values, y_train.values)
    
    # выводим результаты
    if not mute:
        print('='*20 + ' Stage 1: train ' + '='*20 + '\n')
        print('Best estimator: %s' % (gsearch.best_estimator_))
        print('Best train score: %f %s using %s\n' % (abs(gsearch.best_score_), gsearch.scorer_, gsearch.best_params_))

    # выбираем лучшие результаты
    stage1_best_params = []
    means = gsearch.cv_results_['mean_test_score']
    stds = gsearch.cv_results_['std_test_score']
    params = gsearch.cv_results_['params']
    ranks = gsearch.cv_results_['rank_test_score']
    for mean, stdev, param, rank in zip(means, stds, params, ranks):
        # print('%s. %f (%f) with: %r' % (rank, abs(mean), stdev, param))
        if rank < (result + 1):
            stage1_best_params.append({'rank': rank, 'param': param})

    # сортируем параметры по ранку
    stage1_sorted_params = []
    for param in sorted(stage1_best_params, key=(lambda x: x['rank'])):
        stage1_sorted_params.append(param['param'])

    ###
    ### STAGE 2: валидируем модель на валидационных данных
    ###
    
    # проверяем лучшие параметры по валидационной выборке
    stage2_valid_score = -np.inf if scoring_greater_is_better else np.inf
    stage2_valid_param = {}
    stage2_y_valid_pred = []
    stage2_y_valid_probas = []
    stage2_valid_t_score = 0
    stage2_valid_t_threshold = 0

    for param in stage1_sorted_params:

        stage2_model = estimator(**param).fit(X_train.values, y_train.values)

        if len(threshold) == 0:
            y_valid_pred = stage2_model.predict(X_valid.values)
        else:
            y_valid_probas = stage2_model.predict_proba(X_valid.values)
            t_score = -np.inf
            t_threshold = 0
            for t in threshold:
                t_y_valid_pred = (y_valid_probas[:,1] > t).astype(int)
                t_s = scoring_f(y_valid, t_y_valid_pred)
                if t_s > t_score:
                    t_score = t_s
                    t_threshold = t
                    y_valid_pred = t_y_valid_pred

        stage2_score = scoring_f(y_valid, y_valid_pred)

        if (stage2_score > stage2_valid_score) and scoring_greater_is_better:
            stage2_valid_score = stage2_score
            stage2_valid_param = param
            stage2_y_valid_pred = y_valid_pred
            if len(threshold) > 0:
                stage2_y_valid_probas = y_valid_probas
                stage2_valid_t_score = t_score
                stage2_valid_t_threshold = t_threshold
        if (stage2_score < stage2_valid_score) and not scoring_greater_is_better:
            stage2_valid_score = stage2_score
            stage2_valid_param = param
            stage2_y_valid_pred = y_valid_pred
    
    # выводим результаты
    if not mute:

        print('='*20 + ' Stage 2: valid ' + '='*20 + '\n')
        print('Best valid score: %f using %s\n' % (stage2_valid_score, stage2_valid_param))

        # если модель бинарной классификации не сбалансирована
        if len(threshold) > 0:
            print('Best threshold score: %f using %f\n' % (stage2_valid_t_score, stage2_valid_t_threshold))
            skplt.metrics.plot_precision_recall(y_valid.values, stage2_y_valid_probas)
            plt.show()

        # если модель является классификацией
        if scoring_greater_is_better:
            print(classification_report(y_valid, stage2_y_valid_pred))
            skplt.metrics.plot_confusion_matrix(y_valid, stage2_y_valid_pred, normalize=False)
            plt.show()

        # вызываем функцию визуализации кривых обучения
        if learning_curves_dots != 0:
            print('='*20 + ' Stage 2: learning curves ' + '='*21)
            learning_curves(
                estimator(**stage2_valid_param),
                X_train, y_train, X_valid, y_valid,
                learning_curves_dots=learning_curves_dots,
                scoring=scoring,
                scoring_f=scoring_f,
                cv=cv,
                threshold=stage2_valid_t_threshold
            )

    ###
    ### STAGE 3: делаем предикт по тестовым данным
    ###

    # обучаем модель на лучших параметрах с предыдущего шага
    stage3_model = estimator(**stage2_valid_param).fit(X_train, y_train)
    
    # если тестовой выборки нет, то значим просто завершаем работу и возвращаем модель
    if len(X_test) == 0:
        return { 'model': stage3_model, 'y_valid_pred': stage2_y_valid_pred }
    
    # предсказываем целевую переменную по X_test
    if len(threshold) == 0:
        y_test_pred = stage3_model.predict(X_test)
    else:
        y_test_probas = stage3_model.predict_proba(X_test)
        y_test_pred = (y_test_probas[:,1] > stage2_valid_t_threshold).astype(int)
    
    # если нет результатов для сверки, то завершаем работу и возвращаем модель и предикт
    if len(y_test) == 0:
        return { 'model': stage3_model, 'y_pred': y_test_pred, 'y_valid_pred': stage2_y_valid_pred }

    # если есть данные для сверки, то считаем скоринг
    stage3_score = scoring_f(y_test, y_test_pred)

    # выводим результаты
    if not mute:
        print('='*20 + ' Stage 3: test ' + '='*21 + '\n')
        print('Best test score: %f using %s\n' % (stage3_score, stage2_valid_param))
    
    # возвращаем модель и предикт
    return { 'model': stage3_model, 'y_pred': y_test_pred, 'y_valid_pred': stage2_y_valid_pred }

# делаем визуализацию кривой обучения на основе модели и данных для обучения
def learning_curves(
    model,
    X_train, y_train, X_valid, y_valid,
    learning_curves_dots=10,
    scoring='accuracy',
    scoring_f=accuracy_score,
    cv=5,
    threshold=0
):

    train_errors, valid_errors = [], []
    x = []

    # определяем learning_curves_step исходя из желаемого числа точек на графике
    learning_curves_step = math.ceil(len(X_train) / learning_curves_dots)
    
    # собираем информацию об обучении модели и ошибках
    for m in range(cv, len(X_train)+1, learning_curves_step):

        model.fit(X_train[:m].values, y_train[:m].values)

        if threshold == 0:
            y_train_pred = model.predict(X_train[:m].values)
            y_valid_pred = model.predict(X_valid.values)
        else:
            y_train_probas = model.predict_proba(X_train[:m].values)
            y_train_pred = (y_train_probas[:,1] > threshold).astype(int)
            y_valid_probas = model.predict_proba(X_valid.values)
            y_valid_pred = (y_valid_probas[:,1] > threshold).astype(int)

        train_errors.append(scoring_f(y_train[:m], y_train_pred))
        valid_errors.append(scoring_f(y_valid, y_valid_pred))
        x.append(m)

    train_errors = train_errors if scoring != 'neg_mean_squared_error' else np.sqrt(train_errors)
    valid_errors = valid_errors if scoring != 'neg_mean_squared_error' else np.sqrt(valid_errors)

    # выводим график кривых обучения
    visualize_learning_curves(train_errors, valid_errors, x, scoring_name=scoring)
