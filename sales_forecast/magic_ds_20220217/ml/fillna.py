import pandas as pd
import numpy as np

# заполняем пропуски в списке колонок средним значением
def fillna_mean(df, columns):

    df0 = df.copy()
    for c in columns:
        df0[c] = df[c].fillna( df[c].mean() )
    return df0

# заполняем пропуски в списке колонок медианным значением
def fillna_median(df, columns):

    df0 = df.copy()
    for c in columns:
        df0[c] = df[c].fillna( df[c].median() )
    return df0

# заполняем отрицательные значения в списке колонок медианным
def fillna_negative(df, columns):

    df0 = df.copy()
    for c in columns:
        median = df[c].median()
        df0[c] = df[c].map(lambda x: median if x<0 else x)
    return df0

# заполняем пропуски в списке колонок модовым значением
def fillna_mode(df, columns):

    df0 = df.copy()
    for c in columns:
        df0[c] = df[c].fillna( df[c].mode().values[0] )
    return df0

# заполняем пропуски в списке колонок переданным значением
def fillna_value(df, columns, value=-1):

    df0 = df.copy()
    for c in columns:
        df0[c] = df[c].fillna( value )
    return df0

# заполняем пропуски в списке колонок NaN
def fillna_nan(df, columns):

    df0 = df.copy()
    for c in columns:
        df0[c] = df[c].fillna( np.nan )
    return df0
