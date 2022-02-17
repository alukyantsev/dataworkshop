import pandas as pd
import numpy as np
from . gap import *

# удаляет колонки
def drop_features(df, columns):

    df0 = df.copy()
    df0 = df0.drop(columns = columns)
    return df0

# удаляет колонки с пропусками более limit %
def drop_gap(df, limit=50):

    gap_df = gap_info(df)
    gap_columns = list(gap_df[ gap_df['% of Total Values'] > limit ].index)
    print('We will remove %d columns with limit %d%%.' % (len(gap_columns), limit))
    df0 = df.copy()
    df0 = df0.drop(columns = gap_columns)
    return df0

# находим сильно коррелирующие признаки и удаляем их
def drop_corr(df, limit=0.98):

    df_corr = df.corr()
    columns_corr = []
    index_corr = []
    for column in list(df_corr.columns):
        for index in list(df_corr.index):
            if (column != index) and (column not in index_corr) and (index not in columns_corr):
                if abs(df_corr[column][index] > limit):
                    columns_corr.append(column)
                    index_corr.append(index)
    columns_corr = set(columns_corr)
    index_corr = set(index_corr)
    
    df0 = df.copy()
    df0 = df0.drop(columns = columns_corr)
    return df0
