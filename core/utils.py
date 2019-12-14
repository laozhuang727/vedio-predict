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

if __name__ == '__main__':
    print_lgb_importance()
