import pandas as pd
import numpy as np
import time

# преобразуем список колонок типа uint8 в int64 (возникают в результате one-hot-encoding)
def transform_int64(df, columns=[]):

    df0 = df.copy()
    for c in columns if len(columns) > 0 else list(df.select_dtypes(include=['uint8']).columns):
        df0[c] = df0[c].astype('int64')
    return df0

# преобразуем список колонок типа number в float64
def transform_float64(df, columns=[]):

    df0 = df.copy()
    for c in columns if len(columns) > 0 else list(df.select_dtypes(include=['number']).columns):
        df0[c] = df0[c].astype('float64')
    return df0

# фильтруем колонки, исключая из них exclude
# используется для выборки всех колонок определенного типа за исключением некоторых
# далее столбцы могут передаваться на нормализацию и стандартизацию
def filter_columns(df, columns=[], exclude=[], dtype=''):

    if len(columns) == 0:
        columns = df.select_dtypes(dtype).columns if len(dtype) > 0 else df.columns
    columns = [x for x in columns if x not in exclude]
    return columns

# переименовываем колонки в более привычный вид
def rename_columns(df):

    columns = [x.replace(" ", "_").lower() for x in df.columns]
    return columns

def save_csv(df, path='', name=''):

    df.to_csv(path + name + '_' + str(round(time.time())) + '.csv', index=False)
