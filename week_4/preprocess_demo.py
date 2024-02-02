# just import the libraries you need
# you are not showing UI so no matplotlib or seaborn
# nor exploring the dataset so no info() or describe() type calls

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

# every python file has a name
# if this file is running as a script (python3 preprocess_demo.py) then it's name is __main__
# if imported as a module then it's name is the name of the module, in this case preprocess_demo
# the code below only runs when running as a script
if __name__== "__main__":
    #load raw t-shirt order
    df = ut.generate_tshirt_order(100,100,100,dups=100, percent_nans=0.2)
    df.iloc[1,3]='"-,.."'

    # this is the hand coded bit for nominal cat var
    vals ={'t_shirt_size': {'large': 2, 'medium': 1, 'small': 0}}
    
    #run a pipeline of transforms, note all functions are from ut namespace
    df=df.pipe(ut.impute_NaNs).pipe(ut.ps_lower_strip).pipe(ut.ps_replace_punctuation,['name']).pipe(ut.remove_duplicates,['name']).pipe(ut.cat_ordinal,['t_shirt_size'],vals).pipe(ut.drop_no_variance_columns).pipe(ut.scale).pipe(ut.cat_getdummies, ['t_shirt_color']).pipe(ut.drop_correlated_columns)

    df.to_feather('./script.feather')