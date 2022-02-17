import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler, PolynomialFeatures
from sklearn.utils import resample, shuffle
from imblearn.over_sampling import SMOTE, ADASYN
from . fillna import *

# считаем логарифм списка колонок
def transform_log(df, columns):

    df0 = df.copy()
    for c in columns:
        df0[c] = df[c].map(lambda x: np.log(x) if x > 0 else np.nan) \
                      .replace({np.inf: np.nan, -np.inf: np.nan})
    
    df0 = fillna_median(df0, columns)
    return df0

# проводим нормализацию списка колонок
def transform_normalize(df, columns, method='yeo-johnson'):

    transformer = PowerTransformer(method=method, standardize=False)
    df0 = df.copy()
    for c in columns:
        df0[c] = transformer.fit_transform(df[c].values.reshape(df.shape[0], -1))
    return df0

# проводим стандартизацию списка колонок
def transform_scaler(df, columns):

    scaler = StandardScaler()
    df0 = df.copy()
    df0[columns] = scaler.fit_transform(df[columns])
    return df0

# проводим логарифмизацию, нормализацию, а потом стандартизацию списка колонок
#
# method='yeo-johnson' - works with positive and negative values
# method='box-cox' - only works with strictly positive values
#
def transform_features(df, columns_log=[], columns_normalize=[], columns_scaler=[], method='yeo-johnson'):

    df0 = transform_log(df, columns_log) if len(columns_log) > 0 else df
    df1 = transform_normalize(df0, columns_normalize) if len(columns_normalize) > 0 else df0
    df2 = transform_scaler(df1, columns_scaler) if len(columns_scaler) > 0 else df1
    return df2

# делаем полиномиальное преобразование
def transform_poly(X, degree=2):

    X0 = X.copy()
    poly = PolynomialFeatures(degree=degree)
    X0 = poly.fit_transform(X0)
    return X0

# делает oversampling для несбалансированного датасета
def transform_oversample(X_train, y_train, major_value=np.nan, minor_value=np.nan, random_state=42):

    if np.isnan(major_value):
        major_value = list(y_train.value_counts().sort_values(ascending=False).index)[0]
    if np.isnan(minor_value):
        minor_value = list(y_train.value_counts().sort_values(ascending=True).index)[0]

    target = 'target'
    df0 = pd.concat( [X_train, pd.Series(y_train, name=target)], axis=1 )
    major = df0[ df0[target] == major_value ]
    df0_oversampled = major

    for i in list(y_train.value_counts().sort_values(ascending=True).index)[:-1]:
        minor = df0[ df0[target] == i ]
        minor_oversampled = resample(minor, replace=True, n_samples=len(major), random_state=random_state)
        df0_oversampled = pd.concat([df0_oversampled, minor_oversampled])

    df0_oversampled = shuffle(df0_oversampled, random_state=random_state)

    return { 'X_train': df0_oversampled.drop(target, axis=1), 'y_train': df0_oversampled[target] }

# делает undersampling для несбалансированного датасета
def transform_undersample(X_train, y_train, major_value=np.nan, minor_value=np.nan, random_state=42):

    if np.isnan(major_value):
        major_value = list(y_train.value_counts().sort_values(ascending=False).index)[0]
    if np.isnan(minor_value):
        minor_value = list(y_train.value_counts().sort_values(ascending=True).index)[0]

    target = 'target'
    df0 = pd.concat( [X_train, pd.Series(y_train, name=target)], axis=1 )
    minor = df0[ df0[target] == minor_value ]
    df0_undersampled = minor

    for i in list(y_train.value_counts().sort_values(ascending=False).index)[:-1]:
        major = df0[ df0[target] == i ]
        major_oversampled = resample(major, replace=True, n_samples=len(minor), random_state=random_state)
        df0_undersampled = pd.concat([df0_undersampled, major_oversampled])

    df0_undersampled = shuffle(df0_undersampled, random_state=random_state)

    return { 'X_train': df0_undersampled.drop(target, axis=1), 'y_train': df0_undersampled[target] }

# делает преобразование SMOTE для несбалансированного датасета
def transform_smote(X_train, y_train, random_state=42):    
    
    sm = SMOTE(random_state=random_state)
    X0, y0 = sm.fit_resample(X_train, y_train)
    
    return { 'X_train': X0, 'y_train': y0 }

# делает преобразование ADASYN для несбалансированного датасета
def transform_adasyn(X_train, y_train, sampling_strategy='auto', random_state=42):    
    
    ada = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
    X0, y0 = ada.fit_resample(X_train, y_train)
    
    return { 'X_train': X0, 'y_train': y0 }
