# -*- coding:utf-8 -*-
"""

Author:
    ruiyan zry,15617240@qq.com

"""

import pandas as pd
import numpy as np
import time, datetime
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from tabulate import tabulate

from core.utils import timeit, print_lgb_importance

path = "/Users/ryan/Downloads/data/"
path_sub = path + 'sub/'
path_npy = path + 'npy/'
path_data = path + 'raw/'
path_model = path + 'model/'
path_result = path + 'result/'
path_pickle = path + 'pickle/'
path_profile = path + 'profile/'

debug_small = True
if debug_small:
    train = pd.read_pickle(path_pickle + 'train_small.pickle')
    test = pd.read_pickle(path_pickle + 'test_small.pickle')
    app = pd.read_pickle(path_pickle + 'app_small.pickle')
    user = pd.read_pickle(path_pickle + 'user_small.pickle')
else:
    train = pd.read_pickle(path_pickle + 'train.pickle')
    test = pd.read_pickle(path_pickle + 'test.pickle')
    app = pd.read_pickle(path_pickle + 'app.pickle')
    user = pd.read_pickle(path_pickle + 'user.pickle')

# print(train[0:30].to_string())

print(tabulate(train.head(20), headers='keys', tablefmt='psql'))

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
print(missing_data.head(20))


def find_top1_in_group(x):
    if x.count() <= 0:
        return np.nan
    return x.value_counts().index[0]


all_data['guid'] = all_data.groupby('deviceid')['guid'].transform(find_top1_in_group)
all_data['guid'] = all_data['guid'].fillna(all_data['guid'].value_counts().idxmax())

print(tabulate(all_data.head(20), headers='keys', tablefmt='psql'))
