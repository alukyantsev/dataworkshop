import pandas as pd
import numpy as np
from tqdm import tqdm
from . model_scoring import *

# подбирает коэффициенты для списка моделей на основе валидационных данных
# bagging_coef(y_valid.values, [y_valid_xgb_norma, y_valid_ctb_norma, y_valid_lgb_norma], scoring_f=f1_score)
def bagging_coef(y_true, y_preds, scoring_f=accuracy_score, range_from=0, range_max=0.5, range_step=0.1):

    count_range = 0
    res = pd.DataFrame(columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'threshold', 'score'])
    
    t_from, t_max = [], []
    for t in range(9):
        if t < len(y_preds):
            t_from.append(range_from + range_step)
            t_max.append(range_max + range_step)
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
