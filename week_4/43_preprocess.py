# just import the libraries you need
# you are not showing UI so no matplotlib or seaborn
# nor exploring the dataset so no info() or describe() calls

import pandas as pd
import numpy as np

#the following gives access to utils folder
#where utils package stores shared code
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.getcwd(),
                  os.pardir)
)

#only add it once
if (PROJECT_ROOT not in sys.path):
    sys.path.append(PROJECT_ROOT)

import utils as ut
from sklearn.preprocessing import StandardScaler

#these functions are useful, why not put them in utils?
#all functions here
# def scale(df, features, scaler):
#     '''
#     scales numerical_features using the provided scaler 

#     df: dataframe to operate on
#     features: a list of columns to apply to
#     scaler: function that operates on df's features
#     return: transformed df
#     '''
#     df[features] = scaler.fit_transform(df[features])
#     return df


# def cat_nominal(df, features, order):
#     '''
#     apply a numerical order on nominal features

#     df: dataframe to operate on
#     features: a list of columns to apply to (likely 1)
#     order: custom ordering dictionary, very likely hand generated
#     return: transformed df
#     '''
#     for feat in features:
#         df[feat] = df[feat].map(order)
#     return df


# def cat_getdummies(df, features):
#     '''
#     get dummy vars for each feature

#     df: dataframe to operate on
#     features: a list of columns to apply to
#     return: transformed df
#     '''
#     for feat in features:
#         df = pd.get_dummies(df, columns=[feat])
#     return df

if __name__== "__main__":
    #load raw t-shirt order
    df = ut.generate_tshirt_order()

    # this is the hand coded bit for nominal cat var
    vals = {'large': 2, 'medium': 1, 'small': 0}

    #run a pipeline of transforms
    df_clean = (df.pipe(ut.cat_nominal, ['t_shirt_size'], vals).
            pipe(ut.cat_getdummies, ['t_shirt_color']).
            pipe(ut.scale, ['weight', 't_shirt_size', 'Age'], StandardScaler()))
    
    df_clean.to_feather('./preprocess.feather')