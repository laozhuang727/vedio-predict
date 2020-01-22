# -*- coding:utf-8 -*-
"""

Author:
    ruiyan zry,15617240@qq.com

"""

import pandas as pd
import numpy as np
import time, datetime
import lightgbm as lgb
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from core.utils import timeit, print_lgb_importance

path = "/Users/ryan/Downloads/data/"
path_sub = path + 'sub/'
path_npy = path + 'npy/'
path_data = path + 'raw/'
path_model = path + 'model/'
path_result = path + 'result/'
path_pickle = path + 'pickle/'
path_profile = path + 'profile/'

debug_small = False
if debug_small:
    train = pd.read_pickle(path_pickle + 'train_small.pickle')
    test = pd.read_pickle(path_pickle + 'test_small.pickle')
    app = pd.read_pickle(path_pickle + 'app_small.pickle')
    user = pd.read_pickle(path_pickle + 'user_small.pickle')
else:
    train = pd.read_pickle(path_pickle + 'train.pickle')
    test = pd.read_pickle(path_pickle + 'heatmap.pickle')
    app = pd.read_pickle(path_pickle + 'app.pickle')
    user = pd.read_pickle(path_pickle + 'user.pickle')


# 对数据进行排序
# train = train.sort_values(['deviceid','guid','ts'])
# heatmap = heatmap.sort_values(['deviceid','guid','ts'])

# 查看数据是否存在交集
# train deviceid 104736
# heatmap deviceid 56681
# train&heatmap deviceid 46833
# train guid 104333
# heatmap guid 56861
# train&heatmap guid 46654

@timeit
def analysis_device_guid():
    train_deviceid_set = set(train['deviceid'])
    print('train deviceid', len(train_deviceid_set))
    test_deviceid_set = set(test['deviceid'])
    print('heatmap deviceid', len(test_deviceid_set))
    print('train&heatmap deviceid', len(train_deviceid_set & test_deviceid_set))

    train_guid_set = set(train['guid'])
    print('train guid', len(train_guid_set))
    test_guid_set = set(test['guid'])
    print('heatmap guid', len((set(test['guid']))))
    print('train&heatmap guid', train_guid_set & test_guid_set)

    del train_deviceid_set
    del test_deviceid_set
    del train_guid_set
    del test_guid_set


# analysis_device_guid()


# 时间格式转化 ts
def time_data2(time_sj):
    data_sj = time.localtime(time_sj / 1000)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", data_sj)
    return time_str


train['datetime'] = train['ts'].apply(time_data2)
test['datetime'] = test['ts'].apply(time_data2)

train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

train.loc[(train[['timestamp', 'deviceid', 'newsid']].duplicated()) & (-train['timestamp'].isna()), 'target'] = 0

# 时间范围
# train min: 2019-11-08 00:01:07, train max: 2019-11-10 23:55:51
# train min: 2019-11-11 00:00:00, train max: 2019-11-11 23:59:44
print("train min: {}, train max: {}".format(train['datetime'].min(), train['datetime'].max()))
print("train min: {0}, train max: {1}".format(test['datetime'].min(), test['datetime'].max()))
# 7     0.000000
# 8     0.107774
# 9     0.106327
# 10    0.105583

# 7          11
# 8     3674871
# 9     3743690
# 10    3958109
# 11    3653592


train['days'] = train['datetime'].dt.day
test['days'] = test['datetime'].dt.day

train['flag'] = train['days']
test['flag'] = 11

# 8 9 10 11
data = pd.concat([train, test], axis=0, sort=False)
del train, test

data = data.sort_values(['deviceid', 'ts']).reset_index().drop('index', axis=1)

# 把相隔广告曝光相隔时间较短的数据视为同一个事件，这里暂取间隔为3min
# rank按时间排序同一个事件中每条数据发生的前后关系
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


# 小时信息
data['hour'] = data['datetime'].dt.hour
data['minute'] = data['datetime'].dt.minute
data['hour_min_time'] = np.int64(data['hour']) * 60 + np.int64(data['minute'])

# newsid 为空的内容，应该不是日志内容，也就是脏数据
data.loc[~data['newsid'].isna(), 'isLog'] = 1
data.loc[data['newsid'].isna(), 'isLog'] = 0

# gui 缺失值填充, 选相同的deviceid来填
# data['guid'] = data['guid'].fillna('abc')

def find_top1_in_group(x):
    if x.count() <= 0:
        return np.nan
    return x.value_counts().index[0]


data['guid'] = data.groupby('deviceid')['guid'].transform(find_top1_in_group)
data['guid'] = data['guid'].fillna(data['guid'].value_counts().idxmax())



# 构造历史特征 分别统计前一天 guid deviceid 的相关信息
# 8 9 10 11
history_9 = data[data['days'] == 8]
history_10 = data[data['days'] == 9]
history_11 = data[data['days'] == 10]
history_12 = data[data['days'] == 11]
del data
# 61326
# 64766
# 66547
# 41933
# 42546

# 用户的设备id 各天的分布
# print(len(set(history_9['deviceid'])))
# print(len(set(history_10['deviceid'])))
# print(len(set(history_11['deviceid'])))
# print(len(set(history_12['deviceid'])))
# print(len(set(history_9['deviceid']) & set(history_10['deviceid'])))
# print(len(set(history_10['deviceid']) & set(history_11['deviceid'])))
# print(len(set(history_11['deviceid']) & set(history_12['deviceid'])))
print("history_9 device id counts:", history_9['deviceid'].nunique())
print("history_10 device id counts:", history_10['deviceid'].nunique())
print("history_11 device id counts:", history_11['deviceid'].nunique())
print("history_12 device id counts:", history_12['deviceid'].nunique())
print("History_9_10 device appear again: ",
      len(set(history_9['deviceid'].unique()) & set(history_10['deviceid'].unique())))
print("History_10_11 device appear again: ",
      len(set(history_10['deviceid'].unique()) & set(history_11['deviceid'].unique())))
print("History_11_12 device appear again: ",
      len(set(history_11['deviceid'].unique()) & set(history_12['deviceid'].unique())))

# 61277
# 64284
# 66286
# 41796
# 42347

# 用户的注册id 各天的分布，同一个注册id，可能在多个设备上使用，所以设备id数，要大于注册用户id数
# print(len(set(history_9['guid'])))
# print(len(set(history_10['guid'])))
# print(len(set(history_11['guid'])))
# print(len(set(history_12['guid'])))
# print(len(set(history_9['guid']) & set(history_10['guid'])))
# print(len(set(history_10['guid']) & set(history_11['guid'])))
# print(len(set(history_11['guid']) & set(history_12['guid'])))
print("history_9 guid id counts:", history_9['guid'].nunique())
print("history_10 guid id counts:", history_10['guid'].nunique())
print("history_11 guid id counts:", history_11['guid'].nunique())
print("history_12 guid id counts:", history_12['guid'].nunique())
print("history_9_10 guid Appear again: ", len(set(history_9['guid'].unique()) & set(history_10['guid'].unique())))
print("history_10_11 guid Appear again: ", len(set(history_10['guid'].unique()) & set(history_11['guid'].unique())))
print("history_11_12 guid Appear again: ", len(set(history_11['guid'].unique()) & set(history_12['guid'].unique())))

# 640066
# 631547
# 658787
# 345742
# 350542

# 从这个数据上看，同一个newsID，会在几天内多次出现
# print(len(set(history_9['newsid'])))
# print(len(set(history_10['newsid'])))
# print(len(set(history_11['newsid'])))
# print(len(set(history_12['newsid'])))
# print(len(set(history_9['newsid']) & set(history_10['newsid'])))
# print(len(set(history_10['newsid']) & set(history_11['newsid'])))
# print(len(set(history_11['newsid']) & set(history_12['newsid'])))

print("history_9 newsid id counts:", history_9['newsid'].nunique())
print("history_10 newsid id counts:", history_10['newsid'].nunique())
print("history_11 newsid id counts:", history_11['newsid'].nunique())
print("history_12 newsid id counts:", history_12['newsid'].nunique())
print("history_9_10 newsid Appear again: ", len(set(history_9['newsid'].unique()) & set(history_10['newsid'].unique())))
print("history_10_11 newsid Appear again: ",
      len(set(history_10['newsid'].unique()) & set(history_11['newsid'].unique())))
print("history_11_12 newsid Appear again: ",
      len(set(history_11['newsid'].unique()) & set(history_12['newsid'].unique())))


# deviceid guid timestamp ts 时间特征, 这里有特征穿越的问题，在生产上不能使用
# 这里仅仅为了冲分数而做
def get_history_visit_time(data1, date2):
    data1 = data1.sort_values(['ts', 'timestamp'])

    # timestamp：代表改用户点击改视频的时间戳，如果未点击则为NULL
    # ts：视频暴光给用户的时间戳。
    data1['timestamp_ts'] = data1['timestamp'] - data1['ts']
    data1_tmp = data1[data1['target'] == 1].copy()
    del data1
    for col in ['deviceid', 'guid']:
        for ts in ['timestamp_ts']:
            f_tmp = data1_tmp.groupby([col], as_index=False)[ts].agg({
                '{}_{}_max'.format(col, ts): 'max',
                '{}_{}_mean'.format(col, ts): 'mean',
                '{}_{}_min'.format(col, ts): 'min',
                '{}_{}_median'.format(col, ts): 'median'
            })
        date2 = pd.merge(date2, f_tmp, on=[col], how='left', copy=False)

    return date2


history_10 = get_history_visit_time(history_9, history_10)
history_11 = get_history_visit_time(history_10, history_11)
history_12 = get_history_visit_time(history_11, history_12)

data = pd.concat([history_10, history_11], axis=0, sort=False, ignore_index=True)
data = pd.concat([data, history_12], axis=0, sort=False, ignore_index=True)
del history_9, history_10, history_11, history_12

data = data.sort_values('ts')
data['ts_next'] = data.groupby(['deviceid'])['ts'].shift(-1)
data['ts_next_ts'] = data['ts_next'] - data['ts']

def ctr_caculate(click, show):
    if click == None or np.math.isnan(click):
        return 0
    return click/(show+1.0)

# 当前一天内的特征 leak
for col in [['deviceid'], ['guid'], ['newsid']]:
    print("处理当前一天内的特征，当前col:", col)
    days_count_col_name = '{}_days_count'.format('_'.join(col))
    data[days_count_col_name] = data.groupby(['days'] + col)['id'].transform('count')
    data['{}_hours_count'.format('_'.join(col))] = data.groupby(['hour'] + col)['id'].transform('count')

newsid_days_click_count = 'newsid_days_click_count'
data[newsid_days_click_count] = data[data['target'] == 1].groupby(['days'] + col)['newsid'].transform('count')

print(data[newsid_days_click_count])

# netmodel
data['netmodel'] = data['netmodel'].map({'o': 1, 'w': 2, 'g4': 4, 'g3': 3, 'g2': 2})

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


# posobject_col = [i for i in data.sele
# data['pos'] = data['pos']

# 特征离散化
lbl = LabelEncoder()
object_col = [i for i in data.select_dtypes(object).columns if i not in ['id']]
for i in tqdm(object_col):
    data[i] = lbl.fit_transform(data[i].astype(str))

print('train and predict')
X_train = data[data['flag'].isin([9])]
X_valid = data[data['flag'].isin([10])]
X_test = data[data['flag'].isin([11])]

lgb_param = {
    'learning_rate': 0.1,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': -1,
    'seed': 42,
    'boost_from_average': 'false',

    'subsample': 0.8,
    'colsample_bytree': 0.8,
}

feature = [
    'pos', 'netmodel', 'hour', 'minute', 'hour_min_time', 'isLog',
    'deviceid_timestamp_ts_max', 'deviceid_timestamp_ts_mean',
    'deviceid_timestamp_ts_min', 'deviceid_timestamp_ts_median',
    'guid_timestamp_ts_max', 'guid_timestamp_ts_mean',
    'guid_timestamp_ts_min', 'guid_timestamp_ts_median',
    'deviceid_days_count', 'guid_days_count', 'newsid_days_count',
    'deviceid_hours_count', 'guid_hours_count', 'newsid_hours_count',
    # 'deviceid_days_click_count', 'guid_days_click_count', 'newsid_days_click_count',
    # 'deviceid_days_ctr', 'guid_days_ctr', 'newsid_days_ctr',
    'ts_next_ts',
    'newsid', 'app_version', 'device_vendor', 'osversion', 'device_version',
    'ts_before_rank','ts_after_rank','gap_after_int', 'ts_after_group',
    'dist_int',
    'lat_int', 'lng_int'
    # 'personidentification'
]
target = 'target'

lgb_train = lgb.Dataset(X_train[feature].values, X_train[target].values)
lgb_valid = lgb.Dataset(X_valid[feature].values, X_valid[target].values, reference=lgb_train)
lgb_model = lgb.train(lgb_param, lgb_train, num_boost_round=20000, valid_sets=[lgb_train, lgb_valid],
                      early_stopping_rounds=50, verbose_eval=10, feature_name=feature)

p_test = lgb_model.predict(X_valid[feature].values, num_iteration=lgb_model.best_iteration)
xx_score = X_valid[[target]].copy()
xx_score['predict'] = p_test
xx_score = xx_score.sort_values('predict', ascending=False)
xx_score = xx_score.reset_index()
xx_score.loc[xx_score.index <= int(xx_score.shape[0] * 0.103), 'score'] = 1
xx_score['score'] = xx_score['score'].fillna(0)
print(f1_score(xx_score['target'], xx_score['score']))

del lgb_train, lgb_valid
del X_train, X_valid

print_lgb_importance(lgb_model, feature_importance_path=path_result + "feature_importance.csv")

# 没加 newsid 之前的 f1 score
# 0.5129179717875857
# 0.5197833317587095
# 0.6063125458760602
X_train_2 = data[data['flag'].isin([9, 10])]

lgb_train_2 = lgb.Dataset(X_train_2[feature].values, X_train_2[target].values)
lgb_model_2 = lgb.train(lgb_param, lgb_train_2, num_boost_round=lgb_model.best_iteration, valid_sets=[lgb_train_2],
                        verbose_eval=10, feature_name=feature)

p_predict = lgb_model_2.predict(X_test[feature].values)

submit_score = X_test[['id']].copy()
submit_score['predict'] = p_predict
submit_score = submit_score.sort_values('predict', ascending=False)
submit_score = submit_score.reset_index()
submit_score.loc[submit_score.index <= int(submit_score.shape[0] * 0.103), 'target'] = 1
submit_score['target'] = submit_score['target'].fillna(0)

submit_score = submit_score.sort_values('id')
submit_score['target'] = submit_score['target'].astype(int)

# sample = pd.read_csv('./sample.csv')
# sample.columns = ['id', 'non_target']
# submit_score = pd.merge(sample, submit_score, on=['id'], how='left')
#
# submit_score[['id', 'target']].to_csv('./baseline.csv', index=False)

sample = pd.read_csv(path_data + 'sample.csv')
sample.columns = ['id', 'non_target']
submit_score = pd.merge(sample, submit_score, on=['id'], how='left')

submit_score[['id', 'target']].to_csv(path_sub + 'baseline.csv', index=False)
