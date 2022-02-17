import pandas as pd
import numpy as np
import math
import scikitplot as skplt
import matplotlib.pyplot as plt
from tqdm import tqdm
from . model_scoring import *
from . model_linear import *

# подбирает коэффициенты для списка моделей на основе валидационных данных
# stacking_coef(y_valid.values, [y_valid_xgb_norma, y_valid_ctb_norma, y_valid_lgb_norma], scoring_f=f1_score)
# TODO сделать подбор коэффициентов через рекурсию
# TODO сейчас работает в один поток, попробовать распараллелить
def stacking_coef(
    y_true, y_preds,
    scoring_f=accuracy_score, greater_is_better=True,
    range_from=0, range_max=0.5, range_step=0.1,
    threshold=False
):

    count_range = 0
    if threshold:
        res = pd.DataFrame(columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'threshold', 'score'])
    else:
        res = pd.DataFrame(columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'score'])
    
    t_from, t_max = [], []
    for t in range(9):
        if t < len(y_preds):
            t_from.append(range_from)
            t_max.append(range_max+range_step)
        else:
            t_from.append(0)
            t_max.append(range_step)

    for a in tqdm(np.arange(t_from[0], t_max[0], range_step)):
        for b in np.arange(t_from[1], t_max[1], range_step):
            for c in np.arange(t_from[2], t_max[2], range_step):
                for d in np.arange(t_from[3], t_max[3], range_step):
                    for e in np.arange(t_from[4], t_max[4], range_step):
                        for f in np.arange(t_from[5], t_max[5], range_step):
                            for g in np.arange(t_from[6], t_max[6], range_step):
                                for h in np.arange(t_from[7], t_max[7], range_step):
                                    for i in np.arange(t_from[8], t_max[8], range_step):
                                        if a+b+c+d+e+f+g+h+i == 1:
                                            
                                            range_z = np.arange (range_step, 1, range_step) if threshold else [-1]
                                            for z in range_z:

                                                count_range += 1
                                                y_pred = [0]*len(y_preds[0])
                                                y_i = 0
                                                counters = [a, b, c, d, e, f, g, h, i]

                                                for y in y_preds:
                                                    y_pred += y*counters[y_i]
                                                    y_i += 1
                                                if threshold:
                                                    y_pred = (y_pred > z).astype(int)
                                                score = scoring_f(y_true, y_pred)

                                                if threshold:
                                                    r = pd.Series(data={
                                                        'a':a, 'b':b, 'c':c, 'd':d, 'e':e, 'f':f, 'g':g, 'h':h, 'i':i, \
                                                        'threshold':z, 'score':score
                                                    }, name=count_range)
                                                else:
                                                    r = pd.Series(data={
                                                        'a':a, 'b':b, 'c':c, 'd':d, 'e':e, 'f':f, 'g':g, 'h':h, 'i':i, \
                                                        'score':score
                                                    }, name=count_range)
                                                res = res.append(r)

    if greater_is_better:
        return res[ res['score'] == res['score'].max() ]
    else:
        return res[ res['score'] == res['score'].min() ]

# собирает комбинированный предикт на основе предиктов нескольких моделей
def stacking_pred(
    df_stacking, stacking_valid_pred_list, stacking_pred_list,
    y_valid, y_test=[],
    scoring_f=accuracy_score, greater_is_better=True,
    stacking_coef_first=np.nan, mute=False, threshold=False
):

    # вытаскиваем парметры из переданного датафрейма
    stacking_coef_first = stacking_coef_first if not math.isnan(stacking_coef_first) else df_stacking[:1].index[0]
    stacking_drop_list = ['threshold', 'score'] if threshold else ['score']
    stacking_coef_list = df_stacking[ df_stacking.index == stacking_coef_first ]. \
                        drop(stacking_drop_list, axis=1). \
                        values[0][0:len(stacking_valid_pred_list)][np.newaxis, :].T
    if threshold:
        stacking_threshold = df_stacking[ df_stacking.index == stacking_coef_first ]['threshold'].values[0]

    # смотрим какой получился прогноз на валидационных данных
    y_valid_pred = pd.Series( (stacking_valid_pred_list * stacking_coef_list).sum(axis=0) )
    if not mute:
        print('='*20 + ' Valid data ' + '='*21 + '\n')
        if threshold:
            print('Stage 1 valid predict:\n')
            print(y_valid_pred.value_counts().sort_index())
        else:
            print('Best valid score: %f' % scoring_f(y_valid, y_valid_pred))

    if threshold:
        y_valid_pred = (y_valid_pred > stacking_threshold).astype(int)
        if not mute:
            print('\nStage 2 valid threshold %f:\n' % stacking_threshold)
            print(y_valid_pred.value_counts())
            print('\nStage 2 classification report:\n')
            print(classification_report(y_valid, y_valid_pred))
            skplt.metrics.plot_confusion_matrix(y_valid, y_valid_pred, normalize=False)
            plt.show()

    # смотрим на тестовые данные
    y_pred = pd.Series( (stacking_pred_list * stacking_coef_list).sum(axis=0) )
    if not mute:
        if threshold:
            print('='*20 + ' Test data ' + '='*21 + '\n')
            print('Stage 1 test predict:\n')
            print(y_pred.value_counts().sort_index())
        else:
            if len(y_test) > 0:
                print('='*20 + ' Test data ' + '='*21 + '\n')
                print('Best test score: %f' % scoring_f(y_test, y_pred))

    if threshold:
        y_pred = (y_pred > stacking_threshold).astype(int)
        if not mute:
            print('\nStage 2 valid threshold %f:\n' % stacking_threshold)
            print(y_pred.value_counts())
            if len(y_test) > 0:
                print(classification_report(y_test, y_pred))
                skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False)
                plt.show()

    return y_valid_pred, y_pred

# пропускает модели через стекинг и возвращает лучшее значение
def stacking_fit(
    estimator, estimators_list,
    X_train, y_train, X_valid, y_valid,
    X_test=pd.DataFrame(), y_test=pd.Series(dtype='int64'),
    final_estimator=LogisticRegression(),
    scoring_f=accuracy_score, average=''
):
    
    model = estimator(estimators=estimators_list, final_estimator=final_estimator)
    model_score = model.fit(X_train, y_train).score(X_valid, y_valid)
    
    print('Best valid score: %f\n' % model_score)

    if len(X_test) == 0:
        return { 'model': model, 'model_score': model_score }
        
    y_pred = model.predict(X_test)

    if len(y_test) == 0:
        return { 'model': model, 'model_score': model_score, 'y_pred': y_pred }

    else:
        
        if len(average) == 0:
            model_score = scoring_f(y_test, y_pred)
        else:
            model_score = scoring_f(y_test, y_pred, average=average)

        print('Best test score: %f\n' % model_score)
        return { 'model': model, 'model_score': model_score, 'y_pred': y_pred }
