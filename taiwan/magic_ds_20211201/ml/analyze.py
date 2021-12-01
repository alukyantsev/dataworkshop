import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew, shapiro, normaltest
from . visualize import *

# анализируем распределение данных в списке колонок
def analyze_unique(df, columns=[]):

    for c in columns if len(columns) > 0 else df.columns:
        print('='*20 + ' ' + c + ' (' + str(df[c].nunique()) + ' unique) ' + '='*20)
        value_counts_c = df[c].value_counts()
        print(value_counts_c, '\n')
        if len(columns) == 1:
            return value_counts_c

# анализируем как влияют колонки из списка на целевую переменную
def analyze_target(df, target, columns=[]):

    # функция группировки по признаку с расчетом среднего значения целевой переменной
    mean_target = lambda f: df[[f, target]][~df[target].isnull()].groupby(f, as_index=False).mean().sort_values(by=f, ascending=True)

    for c in columns if len(columns) > 0 else df.columns:
        if c != target:
            print('='*20 + ' ' + c + ' ' + '='*20)
            mean_target_c = mean_target(c)
            print(mean_target_c, '\n')
            if len(columns) == 1:
                return mean_target_c

# сводная информация по корреляциям
def analyze_corr(df):

    return df.corr()

# анализируем список колонок на нормализацию
def analyze_normal(df, columns=[], skew_score=0.5):

    abnormal_list = []
    for c in columns if len(columns) > 0 else df.columns:
        df_skew = skew(df[c])
        print('='*20 + ' ' + c + ' ' + '='*20)
        visualize_features_hist(df, c)
        print('mean : ', np.mean(df[c]))
        print('var  : ', np.var(df[c]))
        print('skew : ', df_skew)
        print('kurt : ', kurtosis(df[c]))
        print('shapiro : ', shapiro(df[c]))
        print('normaltest : ', normaltest(df[c]))
        print('\n')
        if abs(df_skew) > skew_score:
            abnormal_list.append(c)

    return abnormal_list
