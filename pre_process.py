# -*- coding:utf-8 -*-
"""

Author:
    ruiyan zry,15617240@qq.com

"""
import os

import vaex
from sklearn.metrics import log_loss, roc_auc_score, f1_score
from tensorflow.python.keras import backend as K

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from deepctr.models import DeepFM
from deepctr.inputs import SparseFeat, DenseFeat, get_feature_names, VarLenSparseFeat
from tensorflow.python import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

path = "/Users/ryan/Downloads/data/"
path_sub = path + 'sub/'
path_npy = path + 'npy/'
path_data = path + 'raw/'
path_model = path + 'model/'
path_result = path + 'result/'
path_pickle = path + 'pickle/'
hdf5_pickle = path + 'hdf5/'
path_profile = path + 'profile/'
import vaex as vx

debug_samll = True
debug_nrow = 10000


def build_pickle():
    if debug_samll:
        train = pd.read_csv(path_data + 'train.csv', nrows=debug_nrow)
        train.to_pickle(path_pickle + "train_small.pickle")
        print("success build train_small.pickle")

        test = pd.read_csv(path_data + 'test.csv', nrows=debug_nrow)
        test.to_pickle(path_pickle + "test_small.pickle")
        print("success build test_small.pickle")

        app = pd.read_csv(path_data + 'app.csv', nrows=debug_nrow)
        app.to_pickle(path_pickle + "app_small.pickle")
        print("success build app_small.pickle")

        user = pd.read_csv(path_data + 'user.csv', nrows=debug_nrow)
        user.to_pickle(path_pickle + "user_small.pickle")
        print("success build user_small.pickle")
    else:
        train = pd.read_csv(path_data + 'train.csv')
        train.to_pickle(path_pickle + "train.pickle")
        print("success build train.pickle")

        test = pd.read_csv(path_data + 'test.csv')
        test.to_pickle(path_pickle + "test.pickle")
        print("success build test.pickle")

        app = pd.read_csv(path_data + 'app.csv')
        app.to_pickle(path_pickle + "app.pickle")
        print("success build app.pickle")

        user = pd.read_csv(path_data + 'user.csv')
        user.to_pickle(path_pickle + "user.pickle")
        print("success build user.pickle")


def np_save():
    import numpy as geek

    a = geek.arange(5)

    # a is printed.
    print("a is:")
    print(a)

    # the array is saved in the file geekfile.npy
    geek.save('/root/train-data/npy/1.npy', a)

    print("the array is saved in the file geekfile.npy")


def varsparsefeature():
    key2index_len = {'applist': 25730, 'tag': 32539, 'outertag': 192}
    max_len = {'applist': 91, 'tag': 197, 'outertag': 2}
    varlen_feature_columns = [VarLenSparseFeat('%s_key' % i, vocabulary_size=key2index_len[i] + 1,
                                               maxlen=max_len[i],
                                               combiner='mean', embedding_dim=8, weight_name='%s_weight' % i) for i in
                              ['applist', 'tag', 'outertag']]

    # varlen_feature_columns = [VarLenSparseFeat('%s_key' % i, vocabulary_size=100,maxlen=100,
    #                                            combiner='mean', embedding_dim=8,weight_name='%s_weight' % i) for i in
    #                           ['applist', 'tag', 'outertag']]
    print(varlen_feature_columns)


def export_hdf5():
    if debug_samll:
        train = vx.open(path_data + 'train_small.csv')
        train.export_hdf5(hdf5_pickle + "train_small.hdf5", progress=True)
        print("success build train_samll.hdf5")

        test = vx.open(path_data + 'test_small.csv')
        test.export_hdf5(hdf5_pickle + "test_small.hdf5", progress=True)
        print("success build test_small.hdf5")
    else:
        train = vx.open(path_data + 'train.csv')
        train.export_hdf5(hdf5_pickle + "train.hdf5", progress=True)
        print("success build train.hdf5")

        test = vx.open(path_data + 'test.csv')
        test.export_hdf5(hdf5_pickle + "test.hdf5", progress=True)
        print("success build test.hdf5")

        app = vx.open(path_data + 'app.csv')
        app.export_hdf5(hdf5_pickle + "app.hdf5", progress=True)
        print("success build app.hdf5")

        user = vx.open(path_data + 'user.csv')
        user.export_hdf5(hdf5_pickle + "user.hdf5", progress=True)
        print("success build user.hdf5")


def test_concat():
    x1, y1, z1 = np.arange(3), np.arange(3, 0, -1), np.arange(10, 13)
    x2, y2, z2 = np.arange(3, 6), np.arange(0, -3, -1), np.arange(13, 16)
    x3, y3, z3 = np.arange(6, 9), np.arange(-3, -6, -1), np.arange(16, 19)
    w1, w2, w3 = np.array(['cat'] * 3), np.array(['dog'] * 3), np.array(['fish'] * 3)
    x = np.concatenate((x1, x2, x3))
    y = np.concatenate((y1, y2, y3))
    z = np.concatenate((z1, z2, z3))
    w = np.concatenate((w1, w2, w3))

    ds = vaex.from_arrays(x=x, y=y, z=z, w=w)
    ds1 = vaex.from_arrays(x=x1, y=y1, z=z1, w=w1)
    print(ds1.head(10))
    ds2 = vaex.from_arrays(x=x2, y=y2, z=z2)

    # 添加一列静态值
    new_np_col = np.full([len(ds2)], fill_value='hello')
    new_np_col = np.full([len(ds2)], fill_value=np.nan)
    ds2['w'] = new_np_col

    print("\n\nds2:")
    print(ds2.head(10))

    ds3 = vaex.from_arrays(x=x3, y=y3, z=z3, w=w3)

    dd = vaex.concat([ds1, ds2])
    ww = ds1.concat(ds2)
    ww = vaex.concat([ds2,ds1])

    print("\n\nww:")
    print(ww.head(10))


if __name__ == '__main__':
    # os.path.join(path_npy)
    # path = os.path.expanduser(path_npy)
    # test_np_save()

    test_concat()
