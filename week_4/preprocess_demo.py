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
from utils.transforms import *

# every python file has a name
# if this file is running as a script (python3 preprocess_demo.py) then it's name is __main__
# if imported as a module then it's name is the name of the module, in this case preprocess_demo
# the code below only runs when running as a script
if __name__== "__main__":
    #load raw t-shirt order
    df = ut.getdata.generate_tshirt_order(100,100,100,dups=100, percent_nans=0.2)

    # this is the hand coded bit for nominal cat var
    vals ={'t_shirt_size': {'large': 2, 'medium': 1, 'small': 0}}

    #run a pipeline of transforms
    df=run_pipeline(df,dup_features=['name'], dummy_features=['t_shirt_color'], ordinal_features=['t_shirt_size'], ordering_dict=vals)

    
    df.to_feather('./script.feather')