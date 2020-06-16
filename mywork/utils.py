#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import joblib
import os

import xgboost as xgb

import numpy as np


class BaseEstimator(object):
    def __init__(self):
        self._model = xgb.XGBRegressor()

    def load_model(self, path):
        self._model.load_model(path)

    def save_model(self, path):
        self._model.save_model(path)

    def fit(self, X, y):
        return self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)


class OneHotEncoder(object):
    def __init__(self, network_type):
        self._init_map()
        self.network_type = network_type

        pass

    def _init_map(self):
        self.ks_map = self.construct_maps(keys=(3, 5, 7))
        self.ex_map = self.construct_maps(keys=(3, 4, 6))
        self.dp_map = self.construct_maps(keys=(2, 3, 4))

    @staticmethod
    def construct_maps(keys):
        d = dict()
        keys = list(set(keys))
        for k in keys:
            if k not in d:
                d[k] = len(list(d.keys()))
        return d

    def _mask_ps_depth(self, ks_list, ex_list, d_list):
        if 'proxyless' in self.network_type:
            d_list = d_list[:-1]
        start = 0
        end = 4
        for d in d_list:
            for j in range(start + d, end):
                ks_list[j] = 0
                ex_list[j] = 0
            start += 4

    def spec2feats(self, ks_list, ex_list, d_list, r):

        self._mask_ps_depth(ks_list, ex_list, d_list)
        len_ = 63 if 'proxyless' in self.network_type else 60
        ks_onehot = [0 for _ in range(len_)]
        ex_onehot = [0 for _ in range(len_)]
        r_onehot = [0 for _ in range(8)]

        for i in range(20):
            start = i * 3
            if ks_list[i] != 0:
                ks_onehot[start + self.ks_map[ks_list[i]]] = 1
            if ex_list[i] != 0:
                ex_onehot[start + self.ex_map[ex_list[i]]] = 1

        if 'proxyless' in self.network_type:
            ks_onehot[60 + self.ks_map[ks_list[i]]] = 1
            ex_onehot[60 + self.ex_map[ex_list[i]]] = 1

        r_onehot[(r - 112) // 16] = 1

        return ks_onehot + ex_onehot + r_onehot


def subdirs(path):
    for entry in os.scandir(path):
        if entry.is_dir():
            yield entry.name


def gen_latency_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--dir',
                        type=str,
                        help="root path of sample set")
    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        help="Inference batch size",
                        default=1)
    parser.add_argument('--save-path', type=str)
    args = parser.parse_args()

    X = []
    y = []
    for entry in subdirs(args.dir):
        curr_path = os.path.join(args.dir, entry)
        dim = json.load(open(os.path.join(curr_path, 'dim.json'), 'r'))
        met_str = 'metrics_{}.json'.format(args.batch_size)
        metrics = json.load(open(os.path.join(curr_path, met_str), 'r'))
        for m in metrics:
            enc = OneHotEncoder(network_type='proxyless')
            result = enc.spec2feats(ks_list=dim['ks'],
                                    ex_list=dim['e'],
                                    d_list=dim['d'],
                                    r=m['img_size'])
            X.append(result)
            y.append(m['lat'])

    print(max(y), min(y))
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(np.array(X),
                                                        np.array(y),
                                                        test_size=0.2)

    est = BaseEstimator()
    est.fit(X_train, y_train)
    est.save_model('./.tmp/lmodel/rtx-2080ti_batch_size_64.lmodel')
    est = BaseEstimator()
    est.load_model('./.tmp/lmodel/rtx-2080ti_batch_size_64.lmodel')
    print(est.predict(X_test[0].reshape(1, -1)), y_test[0])
    from sklearn.metrics import mean_squared_error
    print(mean_squared_error(est.predict(X_test), y_test, squared=False))


if __name__ == '__main__':

    #  gen_latency_model()

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str)
    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()

    history = []
    for file_ in os.listdir(args.dir):
        curr_path = os.path.join(args.dir, file_)
        curr = json.loads(open(curr_path, 'r').read())
        for c in curr:
            history.append(c)

    X = []; y = []
    for curr in history:
        #  dim = curr['dim']
        dim = curr['net_dwe']
        #  cfg = curr['cfg']
        img_size = curr['img_size']
        #  avg_meter = curr['avg_meter']
        avg_meter = curr['accuracy']
        enc = OneHotEncoder(network_type='proxyless')
        result = enc.spec2feats(ks_list=dim['ks'], ex_list=dim['e'], d_list=dim['d'], r=img_size)
        X.append(np.array(result))
        y.append(np.array(avg_meter[1]))

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(np.array(X),
    np.array(y),
    test_size=0.2)
    est = BaseEstimator()
    est.fit(X_train, y_train)
    est.save_model('/home/notreal1995/Master-Thesis/curr/once-for-all/.tmp/lmodel/acc-new.lmodel')
    #  print(X_test[0].reshape((1, -1)).shape)
    #  print(X_test.shape)
    #  print(est.predict(X_test[0].reshape(1, -1)), y_test[0])
    #  est = AccuaryEstimator()
    #  est.fit(X_train, y_train)
    #  est.save_model(args.save_path)
    #  from sklearn.metrics import mean_squared_error
    #  print(mean_squared_error(est.predict(X_test), y_test, squared=False))
