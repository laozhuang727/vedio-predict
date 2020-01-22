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
from scipy.stats import entropy, kurtosis
import time
import gc
pd.set_option('display.max_columns', None)

is_debug = False
path = "/Users/ryan/Downloads/data/"
path_sub = path + 'sub/'
path_npy = path + 'npy/'
path_data = path + 'raw/'
path_model = path + 'model/'
path_result = path + 'result/'
path_pickle = path + 'pickle/'
path_profile = path + 'profile/'

def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
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
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df

print('=============================================== read train ===============================================')
t = time.time()
if is_debug :
    train_df = pd.read_csv(path_data+ 'train_small.csv')
else:
    train_df = pd.read_csv(path_data + 'train.csv')
train_df['date'] = pd.to_datetime(
    train_df['ts'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x / 1000)))
)
train_df['day'] = train_df['date'].dt.day
# 训练集中，day=7的个数为11个，day=8的为3,674,871。 day9，10也是解决40w
# day=7占比不到1/百万，属于异常情况，去掉合理？ 线上的表现又会如何，为啥不是直接删除，这样有点过了
# 这里为啥只是改了day，不去直接改ts和timestamp呢？
train_df.loc[train_df['day'] == 7, 'day'] = 8
train_num = train_df.shape[0]
print(train_df.shape, train_df['target'].mean())
labels = train_df['target'].values
print('runtime:', time.time() - t)

print('=============================================== click data ===============================================')
click_df = train_df[train_df['target'] == 1].sort_values('timestamp').reset_index(drop=True)
print(click_df.shape)
click_df['date'] = pd.to_datetime(
    click_df['timestamp'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x / 1000)))
)
click_df['day'] = click_df['date'].dt.day
click_df.loc[click_df['day'] == 7, 'day'] = 8
click_df['hour'] = click_df['date'].dt.hour
click_df['total_hour'] = click_df['hour'] + 24 * (click_df['day'] - 8)
click_df['exposure_click_gap'] = click_df['timestamp'] - click_df['ts'] # 从曝光到点击的时间差（反应时间）

del train_df['target'], train_df['timestamp'], click_df['date'], click_df['hour']
gc.collect()
print('runtime:', time.time() - t)

print('=============================================== read heatmap ===============================================')
if is_debug:
    test_df = pd.read_csv(path_data + 'test_small.csv')
else:
    test_df = pd.read_csv(path_data + 'heatmap.csv')
test_df['date'] = pd.to_datetime(
    test_df['ts'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x / 1000)))
)
test_df['day'] = test_df['date'].dt.day
# 测试集中，day=10的个数为32个，day=11的为3，653，560占比 1/十万，属于异常情况，去掉合理
test_df.loc[test_df['day'] == 10, 'day'] = 11
print(test_df.shape)

df = pd.concat([train_df, test_df], axis=0, ignore_index=False)
print(df.shape)
del train_df, test_df
gc.collect()
print('runtime:', time.time() - t)


print('=============================================== time proc ===============================================')
df['hour'] = df['date'].dt.hour
df['minute'] = df['date'].dt.minute
df['second'] = df['date'].dt.second
df['day_minute'] = df['hour'] * 60 + df['minute']
df['hour_second'] = df['minute'] * 60 + df['second']
df['day_second'] = df['hour'] * 3600 + df['minute'] * 60 + df['second']
df['total_hour'] = df['hour'] + 24 * (df['day'] - 8)
del df['date']
gc.collect()
print('runtime:', time.time() - t)

print('=============================================== cate enc ===============================================')
df['lng_lat'] = df['lng'].astype('str') + '_' + df['lat'].astype('str') # 经纬度连起来代表确切地理位置
click_df['lng_lat'] = click_df['lng'].astype('str') + '_' + click_df['lat'].astype('str')
cate_cols = [
    'deviceid', 'newsid', 'guid', 'pos', 'app_version', 'device_vendor',
    'netmodel', 'osversion', 'lng', 'lat', 'device_version', 'lng_lat'
]
for f in cate_cols:
    print(f)
    map_dict = dict(zip(df[f].unique(), range(df[f].nunique())))
    df[f] = df[f].map(map_dict).fillna(-1).astype('int32')
    click_df[f] = click_df[f].map(map_dict).fillna(-1).astype('int32')
    df[f + '_count'] = df[f].map(df[f].value_counts())

df = reduce_mem(df)
click_df = reduce_mem(click_df)

print('runtime:', time.time() - t)

print('=============================================== feat eng ===============================================')
sort_df = df.sort_values('ts').reset_index(drop=True)
mode_cols = []
for f in ['deviceid']:
    print('============================================= {} ============================================='.format(f))

    print('************** exposure_click_gap stats **************')
    # 对前一天的样本的所有反应时间进行统计量提取
    tmp = click_df.groupby([f, 'day'], as_index=False)['exposure_click_gap'].agg({
        f + '_prev_day_exposure_click_gap_max': 'max', f + '_prev_day_exposure_click_gap_min': 'min',
        f + '_prev_day_exposure_click_gap_median': 'median', f + '_prev_day_exposure_click_gap_mean': 'mean',
        f + '_prev_day_exposure_click_gap_std': 'std',
        f + '_prev_day_exposure_click_gap_skew': 'skew', f + '_prev_day_exposure_click_gap_kurt': kurtosis,
        f + '_prev_day_exposure_click_gap_q1': lambda x: np.quantile(x, q=0.25),
        f + '_prev_day_exposure_click_gap_q3': lambda x: np.quantile(x, q=0.75)
    })
    tmp[f + '_prev_day_exposure_click_gap_ptp'] = tmp[f + '_prev_day_exposure_click_gap_max'] - tmp[
        f + '_prev_day_exposure_click_gap_min']
    tmp[f + '_prev_day_exposure_click_gap_iqr'] = tmp[f + '_prev_day_exposure_click_gap_q3'] - tmp[
        f + '_prev_day_exposure_click_gap_q1']
    tmp[f + '_prev_day_exposure_click_gap_mean_ratio_std'] = tmp[f + '_prev_day_exposure_click_gap_mean'] / tmp[
        f + '_prev_day_exposure_click_gap_std']
    tmp[f + '_prev_day_exposure_click_gap_mean_ratio_ptp'] = tmp[f + '_prev_day_exposure_click_gap_mean'] / tmp[
        f + '_prev_day_exposure_click_gap_ptp']
    tmp[f + '_prev_day_exposure_click_gap_mean_ratio_iqr'] = tmp[f + '_prev_day_exposure_click_gap_mean'] / tmp[
        f + '_prev_day_exposure_click_gap_iqr']
    # 将day加1再merge即表示前一天，同理减1再merge表示后一天，但会穿越，而且测试集没有后一天
    tmp['day'] += 1
    df = df.merge(tmp, on=[f, 'day'], how='left')
    df = reduce_mem(df)
    print('runtime:', time.time() - t)

    print('************** click_count stats **************')
    # 对前一天的点击次数进行统计
    tmp = click_df.groupby([f, 'day'], as_index=False)['id'].agg({f + '_prev_day_click_count': 'count'})
    tmp['day'] += 1
    df = df.merge(tmp, on=[f, 'day'], how='left')
    df[f + '_prev_day_click_count'] = df[f + '_prev_day_click_count'].fillna(0)

    # 对前一小时的点击次数进行统计（也可以试试统计后一小时的点击次数，但涉及到label的统计量+时间穿越带来的问题是标签泄露）
    # 一小时也可以拓展到两小时、6小时等
    tmp = click_df.groupby([f, 'total_hour'], as_index=False)['id'].agg(
        {'{}_prev_total_hour_click_count'.format(f): 'count'})
    tmp['total_hour'] += 1
    df = df.merge(tmp, on=[f, 'total_hour'], how='left')
    df['{}_prev_total_hour_click_count'.format(f)] = df['{}_prev_total_hour_click_count'.format(f)].fillna(0)
    df = reduce_mem(df)
    print('runtime:', time.time() - t)

    print('************** exposure_count stats **************')
    df = df.merge(df.groupby([f, 'day'], as_index=False)['id'].agg({f + '_day_count': 'count'}), on=[f, 'day'],
                  how='left')
    df = df.merge(df.groupby([f, 'hour'], as_index=False)['id'].agg({f + '_hour_count': 'count'}), on=[f, 'hour'],
                  how='left')
    df = df.merge(df.groupby([f, 'day', 'hour'], as_index=False)['id'].agg({f + '_day_hour_count': 'count'}),
                  on=[f, 'day', 'hour'], how='left')

    # 对前一天的曝光量进行统计
    tmp = df.groupby([f, 'day'], as_index=False)['id'].agg({f + '_prev_day_count': 'count'})
    tmp['day'] += 1
    df = df.merge(tmp, on=[f, 'day'], how='left')
    df[f + '_prev_day_count'] = df[f + '_prev_day_count'].fillna(0)
    # 计算前一天的点击率
    df[f + '_prev_day_ctr'] = df[f + '_prev_day_click_count'] / (
                df[f + '_prev_day_count'] + df[f + '_prev_day_count'].mean())

    # 对前一小时的曝光量进行统计
    tmp = df.groupby([f, 'total_hour'], as_index=False)['id'].agg({f + '_total_hour_count': 'count'})
    tmp['total_hour'] += 1
    df = df.merge(tmp, on=[f, 'total_hour'], how='left')
    df.rename(columns={f + '_total_hour_count': '{}_prev_total_hour_count'.format(f)}, inplace=True)
    df['{}_prev_total_hour_count'.format(f)] = df['{}_prev_total_hour_count'.format(f)].fillna(0)
    # 计算前一小时的点击率
    df['{}_prev_total_hour_ctr'.format(f)] = df['{}_prev_total_hour_click_count'.format(f)] / (
            df['{}_prev_total_hour_count'.format(f)] + df['{}_prev_total_hour_count'.format(f)].mean())
    # 对后一小时的曝光量进行统计（不涉及label的统计量穿越也无妨，但实际应用场景中该类特征无法获取）
    tmp['total_hour'] -= 2
    df = df.merge(tmp, on=[f, 'total_hour'], how='left')
    df.rename(columns={f + '_total_hour_count': '{}_next_total_hour_count'.format(f)}, inplace=True)

    # 对历史总曝光量进行统计（由于每个样本的时间尺度都不同，所以用历史点击率可能比历史曝光好些）
    sort_df[f + '_before_count'] = sort_df.groupby(f).cumcount()
    tmp = sort_df.drop_duplicates([f, 'ts']).reset_index(drop=True)
    df = df.merge(tmp[[f, 'ts', f + '_before_count']], on=[f, 'ts'], how='left')
    df = reduce_mem(df)
    print('runtime:', time.time() - t)

    print('************** exposure_ts_gap stats **************')
    # 第一次曝光到当前的时间差
    tmp = sort_df[[f, 'ts']].drop_duplicates(f, keep='first').reset_index(drop=True)
    tmp.rename(columns={'ts': f + '_first_exposure_ts'}, inplace=True)
    df = df.merge(tmp, on=f, how='left')
    df[f + '_first_exposure_ts_gap'] = df['ts'] - df[f + '_first_exposure_ts']
    del df[f + '_first_exposure_ts']

    # 当前到最后一次曝光的时间差
    tmp = sort_df[[f, 'ts']].drop_duplicates(f, keep='last').reset_index(drop=True)
    tmp.rename(columns={'ts': f + '_last_exposure_ts'}, inplace=True)
    df = df.merge(tmp, on=f, how='left')
    df[f + '_last_exposure_ts_gap'] = df[f + '_last_exposure_ts'] - df['ts']
    del df[f + '_last_exposure_ts']

    prev_gap_cols = []
    next_gap_cols = []
    tmp = sort_df.groupby(f)
    # 前x次、后x次曝光到当前的时间差
    for gap in [1, 2, 3, 5, 8, 10, 20, 30, 50]:
        sort_df['{}_prev{}_exposure_ts_gap'.format(f, gap)] = tmp['ts'].shift(0) - tmp['ts'].shift(gap)
        sort_df['{}_next{}_exposure_ts_gap'.format(f, gap)] = tmp['ts'].shift(-gap) - tmp['ts'].shift(0)
        tmp2 = sort_df[
            [f, 'ts', '{}_prev{}_exposure_ts_gap'.format(f, gap), '{}_next{}_exposure_ts_gap'.format(f, gap)]
        ].drop_duplicates([f, 'ts']).reset_index(drop=True)
        df = df.merge(tmp2, on=[f, 'ts'], how='left')
        prev_gap_cols.append('{}_prev{}_exposure_ts_gap'.format(f, gap))
        next_gap_cols.append('{}_next{}_exposure_ts_gap'.format(f, gap))
    df[f + '_prev_exposure_ts_gap_mean'] = df[prev_gap_cols].mean(axis=1)
    df[f + '_next_exposure_ts_gap_mean'] = df[next_gap_cols].mean(axis=1)
    df = reduce_mem(df)
    del tmp2
    gc.collect()
    print('runtime:', time.time() - t)

    print('************** interactive feat **************')
    # 各离散特征之间的交互
    for col in cate_cols:
        if col == f or col == 'guid':
            continue
        print(col)
        df = df.merge(tmp[col].agg({
            '{}_{}_nunique'.format(f, col): 'nunique',
            '{}_{}_ent'.format(f, col): lambda x: entropy(x.value_counts() / x.shape[0]),  # 熵
            '{}_{}_mode'.format(f, col): lambda x: x.mode()[0]  # 众数（即出现次数最多的item）
        }).reset_index(), on=f, how='left')
        df = df.merge(df.groupby([f, col], as_index=False)['id'].agg({
            '{}_{}_count'.format(f, col): 'count'  # 共现次数
        }), on=[f, col], how='left')
        df['{}_{}_count_ratio'.format(col, f)] = df['{}_{}_count'.format(f, col)] / df[f + '_count']  # 比例偏好
        df['{}_{}_count_ratio'.format(f, col)] = df['{}_{}_count'.format(f, col)] / df[col + '_count']  # 比例偏好
        df['{}_{}_nunique_ratio_{}_count'.format(f, col, f)] = df['{}_{}_nunique'.format(f, col)] / df[f + '_count']
        df['{}_{}_mode_count'.format(f, col)] = df['{}_{}_mode'.format(f, col)].map(
            df['{}_{}_mode'.format(f, col)].value_counts())
        mode_cols.append('{}_{}_mode'.format(f, col))
        print('runtime:', time.time() - t)
    df = reduce_mem(df)
    del tmp
    gc.collect()
    print('runtime:', time.time() - t)
cate_cols.extend(mode_cols)

print('========================================================================================================')
del df['id'], df['ts'], click_df, sort_df
gc.collect()
print(df.shape)

train_df = df[:train_num].reset_index(drop=True)
train_df['label'] = labels
train_df = train_df[train_df['day'] > 8].reset_index(drop=True)
labels = train_df['label'].values
test_df = df[train_num:].reset_index(drop=True)
del df, train_df['label']
gc.collect()

train_idx = train_df[train_df['day'] == 9].index.tolist()
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
    n_estimators=20000,
    num_leaves=127,
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
    num_leaves=127,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=2019
)

print('************** training **************')
clf.fit(
    train_df, labels,
    eval_set=[(train_df, labels)],
    eval_metric='auc',
    categorical_feature=cate_cols,
    early_stopping_rounds=200,
    verbose=50
)
print('runtime:', time.time() - t)

print('************** heatmap predict **************')
if is_debug:
    sub = pd.read_csv(path_data+ 'sample_small.csv')
else:
    sub = pd.read_csv(path_data+ 'sample.csv')
sub['target'] = clf.predict_proba(test_df)[:, 1]
fea_imp_list.append(clf.feature_importances_)
print('runtime:', time.time() - t)

print('=============================================== feat importances ===============================================')
# 特别提醒：注意看下最终的特征重要性靠前的特征，这里面藏着上分点，对这里面某些特征进行拓展可以到前排
fea_imp_dict = dict(zip(train_df.columns.values, np.mean(fea_imp_list, axis=0)))
fea_imp_item = sorted(fea_imp_dict.items(), key=lambda x: x[1], reverse=True)
for f, imp in fea_imp_item:
    print('{} = {}'.format(f, imp))

print(
    '=============================================== threshold search ===============================================')
# f1阈值敏感，所以对阈值做一个简单的迭代搜索
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

