import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

# выборка со всеми известными результатами - делим данные на три части
# данные для обучения 70%, данные для валидации 10% и данные для тестирования 20%
# model_dict = df_split(df_normalize, 'Churn')
#
# выборка с тестовыми данным для предсказания - выделяем выборку test по неизвестному таргету
# делим оставшиеся данные на 2 части: данные для обучения 85%, данные для валидации 15%
# model_dict = df_split(df_normalize, 'Churn', prc_train=85, prc_valid=15, prc_test=0)
#
# обучаем на обучающей выборке, подбираем гиперпараметры на валидационной выборке,
# финальный тест делаем на тестовой выборке
#
# https://medium.com/artificialis/what-is-validation-data-and-what-is-it-used-for-158d685fb921
#
def split_df(df, target, prc_train=70, prc_valid=10, prc_test=20, target_test_value=np.nan, random_state=42):

    if prc_test > 0:
        X = df.drop(target, axis=1)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size = (prc_test + prc_valid) / 100,
            random_state = random_state)
        X_test, X_valid, y_test, y_valid = train_test_split(
            X_test, y_test,
            test_size = prc_valid / (prc_test + prc_valid),
            random_state = random_state)
        return { 
            'X_train': X_train, 'X_valid': X_valid, 'X_test': X_test,
            'y_train': y_train, 'y_valid': y_valid, 'y_test': y_test
        }

    elif prc_test == 0:
        mask1 = df[target].map(lambda x: math.isnan(x))
        mask2 = df[target] == target_test_value
        mask = mask1 if math.isnan(target_test_value) else mask2
        X_test = df[mask].drop(target, axis=1)
        X = df[~mask].drop(target, axis=1)
        y = df[~mask][target]
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y,
            test_size = prc_valid / 100,
            random_state = random_state)
        return { 
            'X_train': X_train, 'X_valid': X_valid, 'X_test': X_test,
            'y_train': y_train, 'y_valid': y_valid
        }

# выводит на экран структуру словаря
def split_dict_struct(df_split):

    print('='*20 + ' Structure ' + '='*21)
    for k in df_split.keys():
        print()
        for l in df_split[k].keys():
            print( '%s -> %s %s' % ( k, l, str(np.array(df_split[k][l]).shape) ) )
            if type(df_split[k][l]) is dict:
                for m in df_split[k][l].keys():
                    print( '%s -> %s -> %s %s' % ( k, l, m, str(np.array(df_split[k][l][m]).shape) ) )
