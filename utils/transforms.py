import pandas as pd
import numpy as np

def scale(df, features, scaler):
    '''
    scales numerical_features using the provided scaler 

    df: dataframe to operate on
    features: a list of columns to apply to (or None if all)
    scaler: function that operates on df's features
    return: transformed df
    '''
    # if features is None:
    #     features=[col for col in df]
    df[features] = scaler.fit_transform(df[features])
    return df


def cat_nominal(df, features, order):
    '''
    apply a numerical order on nominal features

    df: dataframe to operate on
    features: a list of columns to apply to (likely 1)
    order: custom ordering dictionary, very likely hand generated
    return: transformed df
    '''
    for feat in features:
        df[feat] = df[feat].map(order)
    return df


def cat_getdummies(df, features):
    '''
    get dummy vars for each feature

    df: dataframe to operate on
    features: a list of columns to apply to
    return: transformed df
    '''
    for feat in features:
        df = pd.get_dummies(df, columns=[feat])
    return df
