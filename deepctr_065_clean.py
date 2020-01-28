# -*- coding:utf-8 -*-
"""

Author:
    ruiyan zry,15617240@qq.com

"""

# This Python 3 environment comes with many helpful analytics libraries installed
# For example, here's several helpful packages to load in
import datetime

import pandas as pd
import numpy as np

import seaborn as sns
import pandas as pd
import numpy as np
import os

from hyperopt import hp, fmin, rand, tpe, space_eval
from scipy.stats import norm, skew, stats
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
import autopep8
import random

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, auc, accuracy_score, log_loss, f1_score, precision_score, recall_score
from sklearn import preprocessing

from keras.layers import Dropout, Dense, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.models import Sequential
from keras.datasets import mnist
from keras import backend as K
# from keras_radam import RAdam
from tensorflow.python.keras.layers import TimeDistributed, Bidirectional, BatchNormalization
from tqdm import tqdm

import deepctr
from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, DenseFeat, get_feature_names, VarLenSparseFeat

from collections import namedtuple
# from bayes_opt import BayesianOptimization

# 多个变量显示
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'

import warnings

warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['sans-serif']
color = sns.color_palette()
sns.set_style('darkgrid')

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 50)

# path = os.getcwd()

# 简单测试流程联通性
debug_small = True

# 切记这里只能写绝对路径，不然np.save 会报文件不存在的错误
path = "/Users/ryan/Downloads/data/"
path_sub = path + 'sub/'
path_npy = path + 'npy/'
path_data = path + 'raw/'
path_model = path + 'model/'
path_result = path + 'result/'
path_pickle = path + 'pickle/'
path_profile = path + 'profile/'


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df



if debug_small:
    train = pd.read_pickle(path_pickle + 'train_small.pickle')
    test = pd.read_pickle(path_pickle + 'test_small.pickle')
    app = pd.read_pickle(path_pickle + 'app.pickle')
    user = pd.read_pickle(path_pickle + 'user.pickle')
else:
    train = pd.read_pickle(path_pickle + 'train.pickle')
    test = pd.read_pickle(path_pickle + 'heatmap.pickle')
    app = pd.read_pickle(path_pickle + 'app.pickle')
    user = pd.read_pickle(path_pickle + 'user.pickle')

print(test['ts'].isna().value_counts())

# 这里看不大懂
train.loc[(train[['timestamp', 'deviceid', 'newsid']].duplicated()) & (-train['timestamp'].isna()), 'target'] = 0
test['isTest'] = 1
data = pd.concat([train, test], sort=False)
# sort_values 后 index为： 排序前对应的index
# reset_index 后 index为： RangeIndex(start=0, stop=15030273, step=1)
# drop        后 index为： RangeIndex(start=0, stop=15030273, step=1) 感觉drop和不drop没有区别
data = data.sort_values(['deviceid', 'ts']).reset_index().drop('index', axis=1)

data['day'] = data['ts'].apply(
    lambda x: datetime.datetime.utcfromtimestamp(x // 1000).day)
data['hour'] = data['ts'].apply(
    lambda x: datetime.datetime.utcfromtimestamp(x // 1000).hour)
data['minute'] = data['ts'].apply(
    lambda x: datetime.datetime.utcfromtimestamp(x // 1000).minute)
data['second'] = data['ts'].apply(
    lambda x: datetime.datetime.utcfromtimestamp(x // 1000).second)
# time1 倒是合理，把小时和分钟，组成一个新的数值
data['time1'] = np.int64(data['hour']) * 60 + np.int64(data['minute'])

# newsid 为空的内容，应该不是日志内容，也就是脏数据
data.loc[~data['newsid'].isna(), 'isLog'] = 1
data.loc[data['newsid'].isna(), 'isLog'] = 0

# 把相隔广告曝光相隔时间较短的数据视为同一个事件，这里暂取间隔为3min
# rank按时间排序同一个事件中每条数据发生的前后关系
group = data.groupby('deviceid')
data['gap_before'] = group['ts'].shift(0) - group['ts'].shift(1)
data['gap_before'] = data['gap_before'].fillna(3 * 60 * 1000)
INDEX = data[data['gap_before'] > (3 * 60 * 1000 - 1)].index
data['gap_before'] = np.log(data['gap_before'] // 1000 + 1)
data['gap_before_int'] = np.rint(data['gap_before'])
LENGTH = len(INDEX)
ts_group = []
ts_len = []
for i in tqdm(range(1, LENGTH)):
    ts_group += [i - 1] * (INDEX[i] - INDEX[i - 1])
    ts_len += [(INDEX[i] - INDEX[i - 1])] * (INDEX[i] - INDEX[i - 1])
ts_group += [LENGTH - 1] * (len(data) - INDEX[LENGTH - 1])
ts_len += [(len(data) - INDEX[LENGTH - 1])] * (len(data) - INDEX[LENGTH - 1])
data['ts_before_group'] = ts_group
data['ts_before_len'] = ts_len
data['ts_before_rank'] = group['ts'].apply(lambda x: (x).rank())
data['ts_before_rank'] = (data['ts_before_rank'] - 1) / \
                         (data['ts_before_len'] - 1)

group = data.groupby('deviceid')
data['gap_after'] = group['ts'].shift(-1) - group['ts'].shift(0)
data['gap_after'] = data['gap_after'].fillna(3 * 60 * 1000)
INDEX = data[data['gap_after'] > (3 * 60 * 1000 - 1)].index
data['gap_after'] = np.log(data['gap_after'] // 1000 + 1)
data['gap_after_int'] = np.rint(data['gap_after'])
LENGTH = len(INDEX)
ts_group = [0] * (INDEX[0] + 1)
ts_len = [INDEX[0]] * (INDEX[0] + 1)
for i in tqdm(range(1, LENGTH)):
    ts_group += [i] * (INDEX[i] - INDEX[i - 1])
    ts_len += [(INDEX[i] - INDEX[i - 1])] * (INDEX[i] - INDEX[i - 1])
data['ts_after_group'] = ts_group
data['ts_after_len'] = ts_len
data['ts_after_rank'] = group['ts'].apply(lambda x: (-x).rank())
data['ts_after_rank'] = (data['ts_after_rank'] - 1) / (data['ts_after_len'] - 1)

data.loc[data['ts_before_rank'] == np.inf, 'ts_before_rank'] = 0
data.loc[data['ts_after_rank'] == np.inf, 'ts_after_rank'] = 0
data['ts_before_len'] = np.log(data['ts_before_len'] + 1)
data['ts_after_len'] = np.log(data['ts_after_len'] + 1)


min_time = data['ts'].min()
data['timestamp'] -= min_time
data['ts'] -= min_time
data['lat_int'] = np.int64(np.rint(data['lat'] * 100))
data['lng_int'] = np.int64(np.rint(data['lng'] * 100))

group = data[['deviceid', 'lat', 'lng']].groupby('deviceid')
gp = group[['lat', 'lng']].agg(lambda x: stats.mode(x)[0][0]).reset_index()
gp.columns = ['deviceid', 'lat_mode', 'lng_mode']
data = pd.merge(data, gp, on='deviceid', how='left')
data['dist'] = np.log((data['lat'] - data['lat_mode']) **
                      2 + (data['lng'] - data['lng_mode']) ** 2 + 1)
data['dist_int'] = np.rint(data['dist'])
data.loc[data['lat'] != data['lat_mode'], 'isLatSame'] = 0
data.loc[data['lat'] == data['lat_mode'], 'isLatSame'] = 1
data.loc[data['lng'] != data['lng_mode'], 'isLngSame'] = 0
data.loc[data['lng'] == data['lng_mode'], 'isLngSame'] = 1


data = reduce_mem_usage(data)

# data.to_pickle(path_pickle + 'data.pickle')
#
# data = pd.read_pickle(path_pickle + 'data.pickle')
# data = reduce_mem_usage(data)


cate_cols = ['deviceid', 'guid', 'pos', 'app_version',
             'device_vendor', 'netmodel', 'osversion',
             'device_version', 'hour', 'minute', 'second',
             'dist_int',
             'lat_int', 'lng_int', 'gap_before_int', 'ts_before_group',
             'time1', 'gap_after_int', 'ts_after_group',
             ]
drop_cols = ['id', 'target', 'timestamp', 'ts', 'isTest', 'day',
             'lat_mode', 'lng_mode', 'abtarget', 'applist_key',
             'applist_weight', 'tag_key', 'tag_weight', 'outertag_key',
             'outertag_weight', 'newsid']

fillna_cols = [ 'lng', 'lat', 'dist', 'ts_before_rank',
               'ts_after_rank']
data[fillna_cols] = data[fillna_cols].fillna(0)

train_index = data[data['isTest'] != 1].index
train_index, val_index = train_test_split(train_index, test_size=0.2)
test_index = data[data['isTest'] == 1].index
isVarlen = False


# gui 缺失值填充, 选相同的deviceid来填
# data['guid'] = data['guid'].fillna('abc')

def find_top1_in_group(x):
    if x.count() <= 0:
        return np.nan
    return x.value_counts().index[0]


data['guid'] = data.groupby('deviceid')['guid'].transform(find_top1_in_group)
data['guid'] = data['guid'].fillna(data['guid'].value_counts().idxmax())



sparse_features = cate_cols
dense_features = [i for i in data.columns if (
        (i not in cate_cols) & (i not in drop_cols))]
target = 'target'

# 1.Label Encoding for sparse features,and do simple Transformation for dense features
for i in (sparse_features):
    print(i)
    encoder = LabelEncoder()
    data[i] = encoder.fit_transform(data[i])

scaler = StandardScaler()
data[dense_features] = scaler.fit_transform(data[dense_features])
data = reduce_mem_usage(data)

# 2.count #unique features for each sparse field,and record dense feature field name
fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                          for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                          for feat in dense_features]

varlen_feature_columns =[]

dnn_feature_columns = fixlen_feature_columns + varlen_feature_columns
linear_feature_columns = fixlen_feature_columns + varlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 3.generate input data for model
train = data.loc[train_index]
val = data.loc[val_index]
test = data.loc[test_index]

train_model_input = {name: train[name] for name in feature_names}
val_model_input = {name: val[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}


# 4.Define Model, train, predict and evaluate
checkpoint_path = path_model + "cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
# 编译有错，临时去掉embedding_size=8,use_fm=True，编译不过
model = DeepFM(linear_feature_columns, dnn_feature_columns,
               fm_group=dnn_feature_columns, dnn_hidden_units=(256, 256, 256), l2_reg_linear=0.001,
               l2_reg_embedding=0.001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
               dnn_dropout=0.5, dnn_activation='relu', dnn_use_bn=True, task='binary')
try:
    model.load_weights(checkpoint_path);
    print('load weights')
except:
    pass
model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=['accuracy', 'AUC'])
history = model.fit(train_model_input, train[target],
                    batch_size=8192, epochs=5, verbose=1, shuffle=True,
                    callbacks=[cp_callback],
                    validation_data=(val_model_input, val[target]))

data['predict'] = 0
data.loc[train_index, 'predict'] = model.predict(
    train_model_input, batch_size=8192)
data.loc[val_index, 'predict'] = model.predict(
    val_model_input, batch_size=8192)
data.loc[test_index, 'predict'] = model.predict(
    test_model_input, batch_size=8192)

p = 88.5
pred_val = data.loc[val_index, 'predict']
print("val LogLoss", round(log_loss(val[target], pred_val), 4))
threshold_val = round(np.percentile(pred_val, p), 4)
pred_val2 = [1 if i > threshold_val else 0 for i in pred_val]
print("val F1 >%s" % threshold_val, round(
    f1_score(val[target], pred_val2), 4))

pred_train_val = data.loc[data['isTest'] != 1, 'predict']
print("train_val LogLoss", round(log_loss(data.loc[data['isTest'] != 1, 'target'], pred_train_val), 4))
threshold_train_val = round(np.percentile(pred_train_val, p), 4)
pred_train_val2 = [1 if i > threshold_train_val else 0 for i in pred_train_val]
print("train_val F1 >%s" % threshold_train_val, round(
    f1_score(data.loc[data['isTest'] != 1, 'target'], pred_train_val2), 4))

pred_test = data.loc[test_index, 'predict']
threshold_test = round(np.percentile(pred_test, p), 4)
pred_test2 = [1 if i > threshold_test else 0 for i in pred_test]
sub = test[['id', 'target']]
sub['target'] = pred_test2
sub.to_csv(path_sub + 'sub.csv', index=False)
print("successfully save to sub.csv")
