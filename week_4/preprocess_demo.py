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
from sklearn.preprocessing import StandardScaler

# every python file has a name
# if this file is running as a script (python preprocess_demo.py) then it's name is __main__
# if imported as a module then it's name is the name of the module, in this case preprocess_demo
# the code inside the below if only runs when running as a script
if __name__== "__main__":
    #load raw t-shirt order
    df = ut.getdata.generate_tshirt_order()

    # this is the hand coded bit for nominal cat var
    vals = {'large': 2, 'medium': 1, 'small': 0}

    #run a pipeline of transforms
    df_clean = (df.pipe(ut.cat_nominal, ['t_shirt_size'], vals).
            pipe(ut.cat_getdummies, ['t_shirt_color']).
            pipe(ut.scale, ['weight', 't_shirt_size', 'Age'], StandardScaler()))
    
    df_clean.to_feather('./preprocess.feather')