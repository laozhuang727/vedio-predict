# -*- coding:utf-8 -*-
"""

Author:
    ruiyan zry,15617240@qq.com

"""
import time

import vaex
import matplotlib.pyplot as plt
import pandas as pd
import lightgbm as lgb
import numpy as np
import gc


def timeit(func):
    def wrapper():
        start = time.clock()
        func()
        end = time.clock()
        print('used:', end - start)

    return wrapper


def print_lgb_importance(lgb_model, feature_importance_path):
    importance = lgb_model.feature_importance(importance_type='split')
    feature_name = lgb_model.feature_name()
    feature_importance = pd.DataFrame({'feature_name': feature_name, 'importance': importance})
    feature_importance = feature_importance.sort_values(by="importance", ascending=False)
    feature_importance.to_csv(feature_importance_path, index=False)

    plt.figure(figsize=(12, 6))
    lgb.plot_importance(lgb_model, max_num_features=30)
    plt.title("Features Importance")
    plt.show()


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


if __name__ == '__main__':
    print_lgb_importance()
