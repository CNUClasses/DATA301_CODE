# plt.style.use('ggplot')
import pandas as pd
import numpy as np
import random
from random import gauss
# from sklearn.preprocessing import StandardScaler

#to generate people names
import names

PROCESSED_DATA = "./data.feather"

def generate_tshirt_order(numb_small=100, numb_medium=100, numb_large=100):
    '''
    generate a t-shirt order with the above mix of sizes
    use the names module to generate random names associated with each order (see https://pypi.org/project/names/https://pypi.org/project/names/)
    add a color column with a random color
    add a name column with a random name
    add a gender column with a random gender appropriate to the name
    add an Age column with random ages between 8 and 18
    return: a Pandas DataFrame

    assumme: average small person weighs 100lbs
             average medium weighs 140 lbs
             average large weighs 180 lbs
    '''
    #generate a bunch of t-shirts with the following mean,std,numbershirts
    x = np.random.normal(100, 15, numb_small)
    x = np.concatenate((x, np.random.normal(140, 20, numb_medium)))
    x = np.concatenate((x, np.random.normal(180, 30, numb_large)))

    size=np.empty(300, dtype=object)
    size[:numb_small] = 'small'
    size[numb_small:numb_small+numb_medium] = 'medium'
    size[numb_small+numb_medium:numb_small+numb_medium+numb_large] = 'large'

    d = {'weight': x, 't_shirt_size': size}
    df = pd.DataFrame(data=d)

    ts_colors = ['green','blue','orange','red','black']

    df['t_shirt_color'] = np.random.choice(ts_colors, size=numb_small+numb_medium+numb_large)
    df['name'] = "Unknown"
    df.name = df.name.map(lambda x: names.get_full_name())

    #generate an age (integer)
    df['Age'] = np.random.randint(8, 18, len(df))
    return df

NUMB_SAMPLES = 100
RAND_MAX_VAL =10
RAND_MIN_VAL =0
MAX_RISE = 20
RANDOM_SEED=42

def gendata(ns, ):
    '''
    generate dataset for linear regression
    :return: x,y dataset
    '''
    x = [val for val in range(ns)]
    y=[random.random()*(RAND_MAX_VAL-RAND_MIN_VAL)+RAND_MIN_VAL + val + MAX_RISE + RAND_MAX_VAL*gauss(0,1) for val in range(ns)]
    x=np.array(x).reshape(-1,1)
    y=np.array(y).reshape(-1,1)
    return(x,y)
