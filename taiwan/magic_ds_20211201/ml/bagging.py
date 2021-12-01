import pandas as pd
import numpy as np
import math
import scikitplot as skplt
import matplotlib.pyplot as plt
from tqdm import tqdm
from . model_scoring import *

# подбирает коэффициенты для списка моделей на основе валидационных данных
# bagging_coef(y_valid.values, [y_valid_xgb_norma, y_valid_ctb_norma, y_valid_lgb_norma], scoring_f=f1_score)
# TODO сделать подбор коэффициентов через рекурсию
# TODO сейчас работает в один поток, попробовать распараллелить
def bagging_coef(y_true, y_preds, scoring_f=accuracy_score, range_from=0, range_max=0.5, range_step=0.1):

    count_range = 0
    res = pd.DataFrame(columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'threshold', 'score'])
    
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
                                            for z in np.arange (range_step, 1, range_step):

                                                count_range += 1
                                                y_pred = [0]*len(y_preds[0])
                                                y_i = 0
                                                counters = [a, b, c, d, e, f, g, h, i]

                                                for y in y_preds:
                                                    y_pred += y*counters[y_i]
                                                    y_i += 1
                                                y_pred = (y_pred > z).astype(int)
                                                score = scoring_f(y_true, y_pred)

                                                r = pd.Series(data={
                                                    'a':a, 'b':b, 'c':c, 'd':d, 'e':e, 'f':f, 'g':g, 'h':h, 'i':i, \
                                                    'threshold':z, 'score':score
                                                }, name=count_range)
                                                res = res.append(r)

    return res[ res['score'] == res['score'].max() ]

# собирает комбинированный предикт на основе предиктов нескольких моделей
def bagging_pred(
    df_bagging, bagging_valid_pred_list, bagging_pred_list, y_valid, y_test=[],
    bagging_coef_first=np.nan, mute=False
):

    # вытаскиваем парметры из переданного датафрейма
    bagging_coef_first = bagging_coef_first if not math.isnan(bagging_coef_first) else df_bagging[:1].index[0]
    bagging_coef_list = df_bagging[ df_bagging.index == bagging_coef_first ]. \
                        drop(['threshold', 'score'], axis=1). \
                        values[0][0:len(bagging_valid_pred_list)][np.newaxis, :].T
    bagging_threshold = df_bagging[ df_bagging.index == bagging_coef_first ]['threshold'].values[0]

    # смотрим какой получился прогноз на валидационных данных
    y_valid_pred = pd.Series( (bagging_valid_pred_list * bagging_coef_list).sum(axis=0) )
    if not mute:
        print('='*20 + ' Valid data ' + '='*21 + '\n')
        print('Stage 1 valid predict:\n')
        print(y_valid_pred.value_counts().sort_index())

    y_valid_pred = (y_valid_pred > bagging_threshold).astype(int)
    if not mute:
        print('\nStage 2 valid threshold %f:\n' % bagging_threshold)
        print(y_valid_pred.value_counts())
        print('\nStage 2 classification report:\n')
        print(classification_report(y_valid, y_valid_pred))
        skplt.metrics.plot_confusion_matrix(y_valid, y_valid_pred, normalize=False)
        plt.show()

    # смотрим на тестовые данные
    y_pred = pd.Series( (bagging_pred_list * bagging_coef_list).sum(axis=0) )
    if not mute:
        print('='*20 + ' Test data ' + '='*21 + '\n')
        print('Stage 1 test predict:\n')
        print(y_pred.value_counts().sort_index())

    y_pred = (y_pred > bagging_threshold).astype(int)
    if not mute:
        print('\nStage 2 valid threshold %f:\n' % bagging_threshold)
        print(y_pred.value_counts())
        if len(y_test) > 0:
            print(classification_report(y_test, y_pred))
            skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False)
            plt.show()

    return y_valid_pred, y_pred
