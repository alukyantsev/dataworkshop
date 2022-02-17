import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from . drop import *
from . common import *

# кодируем список колонок через one-hot-encoding
def encode_ohe(df, columns):

    df0 = df.copy()
    for c in columns:
        df1 = pd.get_dummies(df0[c], prefix=c, dummy_na=False)
        df0 = pd.concat([df0, df1], axis=1)
    return df0

# кодируем список колонок через one-hot-encoding с удалением старых колонок и трансформацией значений
def encode_ohe_full(df, columns):

    df0 = df.copy()
    df0 = encode_ohe(df0, columns)
    df0 = drop_features(df0, columns)
    df0 = transform_int64(df0)
    return df0

# кодируем список колонок через label-encoding
def encode_le(df, columns, d=[{}]):

    df0 = df.copy()
    i = 0
    for c in columns:

        if len(d[i]) > 0:
            df0[c] = df0[c].map(d[i])
        else:
            label = LabelEncoder()
            label.fit(df[c].drop_duplicates())
            df0[c] = label.transform(df[c])
        
        i += 1

    return df0
