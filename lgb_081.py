# -*- coding:utf-8 -*-
"""

Author:
    ruiyan zry,15617240@qq.com

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import entropy
from gensim.models import Word2Vec
import time
import gc

pd.set_option('display.max_columns', None)

from core.utils import timeit, reduce_mem

##################################################################################################################

# 1、有些说明在前一次开源已经说了，在这里就不说了；
# 2、主要包含四组特征：
# （1）历史信息，即前一天的点击量、曝光量、点击率；
# （2）前x次曝光、后x次曝光到当前的时间差。后x次到当前曝光的时间差是穿越特征，并且是最强的特征；
# （3）二阶交叉特征；
# （4）embedding。
# 3、本来想一开到底的，但是得知15号之后就不让开源了，所以这应该是最后一次了，没机会为大家打开另外两张表了，大家自行发挥吧；
# 4、希望大家都能拿第一名。

##################################################################################################################


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
    train_df = pd.read_pickle(path_pickle + 'train_small.pickle')
    test_df = pd.read_pickle(path_pickle + 'test_small.pickle')
    # app = pd.read_pickle(path_pickle + 'app_small.pickle')
    # user = pd.read_pickle(path_pickle + 'user_small.pickle')
else:
    train_df = pd.read_pickle(path_pickle + 'train.pickle')
    test_df = pd.read_pickle(path_pickle + 'test.pickle')
    # app = pd.read_pickle(path_pickle + 'app.pickle')
    # user = pd.read_pickle(path_pickle + 'user.pickle')

print('=============================================== read train ===============================================')
t = time.time()
# train_df = pd.read_csv('dataset/train.csv')
train_df['date'] = pd.to_datetime(
    train_df['ts'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x / 1000)))
)
train_df['day'] = train_df['date'].dt.day

# 训练集中，day=7的个数为11个，day=8的为3,674,871。 day9，10也是解决40w
# day=7占比不到1/百万，属于异常情况，去掉合理？ 线上的表现又会如何，为啥不是直接删除，这样有点过了
# 这里为啥只是改了day，不去直接改ts和timestamp呢？
train_df.loc[train_df['day'] == 7, 'day'] = 8
train_df['hour'] = train_df['date'].dt.hour
train_df['minute'] = train_df['date'].dt.minute
train_num = train_df.shape[0]
labels = train_df['target'].values
print('runtime:', time.time() - t)

print('=============================================== click data ===============================================')
click_df = train_df[train_df['target'] == 1].sort_values('timestamp').reset_index(drop=True)
click_df['exposure_click_gap'] = click_df['timestamp'] - click_df['ts']
click_df = click_df[click_df['exposure_click_gap'] >= 0].reset_index(drop=True)
click_df['date'] = pd.to_datetime(
    click_df['timestamp'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x / 1000)))
)
click_df['day'] = click_df['date'].dt.day
# 同上对day==7的修改
click_df.loc[click_df['day'] == 7, 'day'] = 8

del train_df['target'], train_df['timestamp']

# 这里为啥要把click_df的这些字段删除呢？
for f in ['date', 'exposure_click_gap', 'timestamp', 'ts', 'target', 'hour', 'minute']:
    del click_df[f]
print('runtime:', time.time() - t)


print('=============================================== read test ===============================================')
test_df['date'] = pd.to_datetime(
    test_df['ts'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x / 1000)))
)
test_df['day'] = test_df['date'].dt.day

# 测试集中，day=10的个数为32个，day=11的为3,653,560占比 1/十万，属于异常情况，去掉合理
test_df.loc[test_df['day'] == 10, 'day'] = 11
test_df['hour'] = test_df['date'].dt.hour
test_df['minute'] = test_df['date'].dt.minute
df = pd.concat([train_df, test_df], axis=0, ignore_index=False)
del train_df, test_df, df['date']
gc.collect()
print('runtime:', time.time() - t)

print('============================================= category encoding =============================================')
df['lng_lat'] = df['lng'].astype('str') + '_' + df['lat'].astype('str')
del df['guid']
click_df['lng_lat'] = click_df['lng'].astype('str') + '_' + click_df['lat'].astype('str')
sort_df = df.sort_values('ts').reset_index(drop=True)
cate_cols = [
    'deviceid', 'newsid', 'pos', 'app_version', 'device_vendor',
    'netmodel', 'osversion', 'device_version', 'lng', 'lat', 'lng_lat'
]
for f in cate_cols:
    print(f)
    map_dict = dict(zip(df[f].unique(), range(df[f].nunique())))
    df[f] = df[f].map(map_dict).fillna(-1).astype('int32')
    click_df[f] = click_df[f].map(map_dict).fillna(-1).astype('int32')
    sort_df[f] = sort_df[f].map(map_dict).fillna(-1).astype('int32')
    df[f + '_count'] = df[f].map(df[f].value_counts())
df = reduce_mem(df)
click_df = reduce_mem(click_df)
sort_df = reduce_mem(sort_df)
print('runtime:', time.time() - t)

print('============================================= feat engineer =============================================')

print('*************************** history stats ***************************')
for f in [
    ['deviceid'],
    ['pos', 'deviceid'],
    # ...
]:
    print('------------------ {} ------------------'.format('_'.join(f)))

    # 对前一天的点击次数进行统计
    tmp = click_df[f + ['day', 'id']].groupby(f + ['day'], as_index=False)['id'].agg(
        {'_'.join(f) + '_prev_day_click_count': 'count'})
    tmp['day'] += 1
    df = df.merge(tmp, on=f + ['day'], how='left')
    df['_'.join(f) + '_prev_day_click_count'] = df['_'.join(f) + '_prev_day_click_count'].fillna(0)
    df.loc[df['day'] == 8, '_'.join(f) + '_prev_day_click_count'] = None

    # 对前一天的曝光量进行统计
    tmp = df[f + ['day', 'id']].groupby(f + ['day'], as_index=False)['id'].agg(
        {'_'.join(f) + '_prev_day_count': 'count'})
    tmp['day'] += 1
    df = df.merge(tmp, on=f + ['day'], how='left')
    df['_'.join(f) + '_prev_day_count'] = df['_'.join(f) + '_prev_day_count'].fillna(0)
    df.loc[df['day'] == 8, '_'.join(f) + '_prev_day_count'] = None

    # 计算前一天的点击率
    df['_'.join(f) + '_prev_day_ctr'] = df['_'.join(f) + '_prev_day_click_count'] / (
            df['_'.join(f) + '_prev_day_count'] + df['_'.join(f) + '_prev_day_count'].mean())

    del tmp
    print('runtime:', time.time() - t)
del click_df
df = reduce_mem(df)

print('*************************** exposure_ts_gap ***************************')
for f in [
    ['deviceid'], ['newsid'], ['lng_lat'],
    ['pos', 'deviceid'], ['pos', 'newsid'], ['pos', 'lng_lat'],
    ['pos', 'deviceid', 'lng_lat'],
    ['netmodel', 'deviceid'],
    ['pos', 'netmodel', 'deviceid'],
    ['netmodel', 'lng_lat'], ['deviceid', 'lng_lat'],
    ['netmodel', 'deviceid', 'lng_lat'], ['pos', 'netmodel', 'lng_lat'],
    ['pos', 'netmodel', 'deviceid', 'lng_lat']
]:
    print('------------------ {} ------------------'.format('_'.join(f)))

    tmp = sort_df[f + ['ts']].groupby(f)
    # 前x次、后x次曝光到当前的时间差
    for gap in [1, 2, 3, 5, 10]:
        sort_df['{}_prev{}_exposure_ts_gap'.format('_'.join(f), gap)] = tmp['ts'].shift(0) - tmp['ts'].shift(gap)
        sort_df['{}_next{}_exposure_ts_gap'.format('_'.join(f), gap)] = tmp['ts'].shift(-gap) - tmp['ts'].shift(0)
        tmp2 = sort_df[
            f + ['ts', '{}_prev{}_exposure_ts_gap'.format('_'.join(f), gap),
                 '{}_next{}_exposure_ts_gap'.format('_'.join(f), gap)]
            ].drop_duplicates(f + ['ts']).reset_index(drop=True)
        df = df.merge(tmp2, on=f + ['ts'], how='left')
        del sort_df['{}_prev{}_exposure_ts_gap'.format('_'.join(f), gap)]
        del sort_df['{}_next{}_exposure_ts_gap'.format('_'.join(f), gap)]
        del tmp2

    del tmp
    df = reduce_mem(df)
    print('runtime:', time.time() - t)
del df['ts']
gc.collect()

print('*************************** cross feat (second order) ***************************')
# 二阶交叉特征，可以继续做更高阶的交叉。
cross_cols = ['deviceid', 'newsid', 'pos', 'netmodel', 'lng_lat']
for f in cross_cols:
    for col in cross_cols:
        if col == f:
            continue
        print('------------------ {} {} ------------------'.format(f, col))
        df = df.merge(df[[f, col]].groupby(f, as_index=False)[col].agg({
            'cross_{}_{}_nunique'.format(f, col): 'nunique',
            'cross_{}_{}_ent'.format(f, col): lambda x: entropy(x.value_counts() / x.shape[0])  # 熵
        }), on=f, how='left')
        if 'cross_{}_{}_count'.format(f, col) not in df.columns.values and 'cross_{}_{}_count'.format(col,
                                                                                                      f) not in df.columns.values:
            df = df.merge(df[[f, col, 'id']].groupby([f, col], as_index=False)['id'].agg({
                'cross_{}_{}_count'.format(f, col): 'count'  # 共现次数
            }), on=[f, col], how='left')
        if 'cross_{}_{}_count_ratio'.format(col, f) not in df.columns.values:
            df['cross_{}_{}_count_ratio'.format(col, f)] = df['cross_{}_{}_count'.format(f, col)] / df[
                f + '_count']  # 比例偏好
        if 'cross_{}_{}_count_ratio'.format(f, col) not in df.columns.values:
            df['cross_{}_{}_count_ratio'.format(f, col)] = df['cross_{}_{}_count'.format(f, col)] / df[
                col + '_count']  # 比例偏好
        df['cross_{}_{}_nunique_ratio_{}_count'.format(f, col, f)] = df['cross_{}_{}_nunique'.format(f, col)] / df[
            f + '_count']
        print('runtime:', time.time() - t)
    df = reduce_mem(df)
del df['id']
gc.collect()

print('*************************** embedding ***************************')


# 之前有个朋友给embedding做了一个我认为非常形象的比喻：
# 在非诚勿扰上面，如果你想了解一个女嘉宾，那么你可以看看她都中意过哪些男嘉宾；
# 反过来也一样，如果你想认识一个男嘉宾，那么你也可以看看他都选过哪些女嘉宾。


def emb(df, f1, f2):
    emb_size = 8
    print('====================================== {} {} ======================================'.format(f1, f2))
    tmp = df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]
    model = Word2Vec(sentences, size=emb_size, window=5, min_count=5, sg=0, hs=1, seed=2019)
    emb_matrix = []
    for seq in sentences:
        vec = []
        for w in seq:
            if w in model:
                vec.append(model[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * emb_size)

    # 为了支持数组多维处理，需要先做一个变换
    emb_matrix = np.array(emb_matrix)

    for i in range(emb_size):
        tmp['{}_{}_emb_{}'.format(f1, f2, i)] = emb_matrix[:, i]
    del model, emb_matrix, sentences
    tmp = reduce_mem(tmp)
    print('runtime:', time.time() - t)
    return tmp


emb_cols = [
    ['deviceid', 'newsid'],
    ['deviceid', 'lng_lat'],
    ['newsid', 'lng_lat'],
    # ...
]
for f1, f2 in emb_cols:
    df = df.merge(emb(sort_df, f1, f2), on=f1, how='left')
    df = df.merge(emb(sort_df, f2, f1), on=f2, how='left')
del sort_df
gc.collect()

print('======================================== prepare train & valid  =============================================')
train_df = df[:train_num].reset_index(drop=True)
test_df = df[train_num:].reset_index(drop=True)
del df
gc.collect()

train_idx = train_df[train_df['day'] < 10].index.tolist()
val_idx = train_df[train_df['day'] == 10].index.tolist()

train_x = train_df.iloc[train_idx].reset_index(drop=True)
train_y = labels[train_idx]
val_x = train_df.iloc[val_idx].reset_index(drop=True)
val_y = labels[val_idx]

del train_x['day'], val_x['day'], train_df['day'], test_df['day']
gc.collect()
print('runtime:', time.time() - t)
print('========================================================================================================')




print('=============================================== training validate ===============================================')
fea_imp_list = []
clf = LGBMClassifier(
    learning_rate=0.01,
    n_estimators=5000,
    num_leaves=255,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=2019,
    metric=None
)

print('************** training **************')
clf.fit(
    train_x, train_y,
    eval_set=[(val_x, val_y)],
    eval_metric='auc',
    categorical_feature=cate_cols,
    early_stopping_rounds=200,
    verbose=50
)
print('runtime:', time.time() - t)

print('************** validate predict **************')
best_rounds = clf.best_iteration_
best_auc = clf.best_score_['valid_0']['auc']
val_pred = clf.predict_proba(val_x)[:, 1]
fea_imp_list.append(clf.feature_importances_)
print('runtime:', time.time() - t)

print('=============================================== training predict ===============================================')
clf = LGBMClassifier(
    learning_rate=0.01,
    n_estimators=best_rounds,
    num_leaves=255,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=2019
)

print('************** training **************')
clf.fit(
    train_df, labels,
    eval_set=[(train_df, labels)],
    categorical_feature=cate_cols,
    verbose=50
)
print('runtime:', time.time() - t)

print('************** test predict **************')
sub = pd.read_csv(path_data + 'sample.csv')
sub['target'] = clf.predict_proba(test_df)[:, 1]
fea_imp_list.append(clf.feature_importances_)
print('runtime:', time.time() - t)

print('=============================================== feat importances ===============================================')
# 特征重要性可以好好看看
fea_imp_dict = dict(zip(train_df.columns.values, np.mean(fea_imp_list, axis=0)))
fea_imp_item = sorted(fea_imp_dict.items(), key=lambda x: x[1], reverse=True)
for f, imp in fea_imp_item:
    print('{} = {}'.format(f, imp))


print('=============================================== threshold search ===============================================')
# f1阈值敏感，所以对阈值做一个简单的迭代搜索。
t0 = 0.05
v = 0.002
best_t = t0
best_f1 = 0
for step in range(201):
    curr_t = t0 + step * v
    y = [1 if x >= curr_t else 0 for x in val_pred]
    curr_f1 = f1_score(val_y, y)
    if curr_f1 > best_f1:
        best_t = curr_t
        best_f1 = curr_f1
        print('step: {}   best threshold: {}   best f1: {}'.format(step, best_t, best_f1))
print('search finish.')

val_pred = [1 if x >= best_t else 0 for x in val_pred]
print('\nbest auc:', best_auc)
print('best f1:', f1_score(val_y, val_pred))
print('validate mean:', np.mean(val_pred))
print('runtime:', time.time() - t)

print('=============================================== sub save ===============================================')
sub.to_csv('sub_prob_{}_{}_{}.csv'.format(best_auc, best_f1, sub['target'].mean()), index=False)
sub['target'] = sub['target'].apply(lambda x: 1 if x >= best_t else 0)
sub.to_csv('sub_{}_{}_{}.csv'.format(best_auc, best_f1, sub['target'].mean()), index=False)
print('runtime:', time.time() - t)
print('finish.')
print('========================================================================================================')
