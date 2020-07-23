#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import joblib
import os

import xgboost as xgb

import numpy as np


class BaseEstimator(object):
    def __init__(self, network_type='proxyless', as_error=False):
        self._model = xgb.XGBRegressor()
        self.network_type = network_type
        self.encoder = OneHotEncoder(network_type='proxyless')
        self.as_error = as_error

    def load_model(self, path):
        self._model.load_model(path)

    def save_model(self, path):
        self._model.save_model(path)

    def fit(self, X, y):
        return self._model.fit(X, y)

    def predict(self, X):
        if self.as_error:
            return 100.0 - self._model.predict(X)
        return self._model.predict(X)

    def predict_with_flatten(self, X):
        ret = []
        len_ = 21 if 'proxyless' in self.network_type else 20
        for x in X:
            ks_list = x[: len_]
            ex_list = x[len_: 2 * len_]
            d_list = x[2 * len_: -1]
            r = [x[-1]]
            enc = self.encoder.spec2feats(ks_list,
                                          ex_list,
                                          d_list,
                                          r[0])

            if self.as_error:
                ret.append(100.0 - self._model.predict(np.array(enc).reshape(1, -1)))
                #  return 100.0 - self._model.predict(np.array(enc).reshape(1, -1))
            else:
                ret.append(self._model.predict(np.array(enc).reshape(1, -1)))
        return np.array(ret)


    def encode2predict(self, X):
        enc = self.encoder.spec2feats(ks_list=X['ks'],
                                      ex_list=X['e'],
                                      d_list=X['d'],
                                      r=X['r'][0])
        if self.as_error:
            return 100.0 - self._model.predict(np.array(enc).reshape(1, -1))

        return self._model.predict(np.array(enc).reshape(1, -1))

    def ir2predict(self, X):
        ret = []
        for x in X:
            enc = self.encoder.ir2feats(x)
            if self.as_error:
                ret.append(100 - self._model.predict(np.array(enc).reshape(1, -1)))
                #  return 100 - self._model.predict(np.array(enc.reshape(1, -1)))
            else:
                ret.append(self._model.predict(np.array(enc).reshape(1, -1)))
            #  return self._model.predict(np.array(enc).reshape(1, -1))
        return np.array(ret)

class OneHotEncoder(object):
    def __init__(self, network_type):
        self._init_map()
        self.network_type = network_type

        pass

    def _init_map(self):
        self.ks_map = self.construct_maps(keys=(3, 5, 7))
        self.ex_map = self.construct_maps(keys=(3, 4, 6))
        self.dp_map = self.construct_maps(keys=(2, 3, 4))
        self.inv_ks_map = {v: k for k, v in self.ks_map.items()}
        self.inv_ex_map = {v: k for k, v in self.ex_map.items()}
        self.inv_dp_map = {v: k for k, v in self.dp_map.items()}

    @staticmethod
    def construct_maps(keys):
        d = dict()
        keys = list(set(keys))
        for k in keys:
            if k not in d:
                d[k] = len(list(d.keys()))
        return d

    def flatten_spec(self, **spec):
        self._mask_ps_depth(spec['ks'], spec['e'], spec['d'])
        return spec['ks'] + spec['e'] + spec['d'] + spec['r']
    

    def _mask_ps_depth(self, ks_list, ex_list, d_list):
        #  if 'proxyless' in self.network_type:
            #  d_list = d_list[:-1]
        start = 0
        end = 4
        for d in d_list[:-1]:
            for j in range(start + d, end):
                ks_list[j] = 0
                ex_list[j] = 0
            start += 4
            end += 4

    def spec2feats(self, ks_list, ex_list, d_list, r):
        
        ks = copy.deepcopy(ks_list)
        ex = copy.deepcopy(ex_list)
        d = copy.deepcopy(d_list)


        self._mask_ps_depth(ks, ex, d)
        len_ = 63 if 'proxyless' in self.network_type else 60
        ks_onehot = [0 for _ in range(len_)]
        ex_onehot = [0 for _ in range(len_)]
        r_onehot = [0 for _ in range(8)]

        num_blocks = 20
        for i in range(num_blocks):
            start = i * 3
            if ks[i] != 0:
                ks_onehot[start + self.ks_map[ks[i]]] = 1
            if ex[i] != 0:
                ex_onehot[start + self.ex_map[ex[i]]] = 1

        if 'proxyless' in self.network_type:
            ks_onehot[60 + self.ks_map[ks[-1]]] = 1
            ex_onehot[60 + self.ex_map[ex[-1]]] = 1

        r_onehot[(r - 112) // 16] = 1

        return ks_onehot + ex_onehot + r_onehot

    def feats2spec(self, enc_list):
        len_ = 63 if 'proxyless' in self.network_type else 60
        ks_onehot = enc_list[:len_]
        ex_onehot = enc_list[len_:2 * len_]
        r_onehot = enc_list[2 * len_:]

        ks_list = []
        ex_list = []
        d_list = []
        for i in range(0, len_, 3):
            stage_ks = np.array(ks_onehot[i:i + 3])
            stage_ex = np.array(ex_onehot[i:i + 3])
            ks_idx = np.argwhere(stage_ks == 1)
            ex_idx = np.argwhere(stage_ex == 1)

            if ks_idx.shape[0] == 0:
                ks_list.append(0)
                ex_list.append(0)
            else:
                ks_list.append(self.inv_ks_map[ks_idx[0][0]])
                ex_list.append(self.inv_ex_map[ex_idx[0][0]])

        stages = 5 if 'proxyless' in self.network_type else 6
        for stage in range(stages):
            is_skip = False
            for d in range(4):
                if ks_list[stage * 4 + d] == 0:
                    d_list.append(d)
                    is_skip = True
                    break
            if not is_skip:
                d_list.append(4)

        if 'proxyless' in self.network_type:
            d_list.append(1)

        r = (np.argwhere(r_onehot)[0][0] * 16) + 112
        return {
            'wid': None,
            'ks': ks_list,
            'e': ex_list,
            'd': d_list,
            'r': [r]
        }

    def specs2ir(self, ks_list, ex_list, d_list, r):

        #  self._mask_ps_depth(ks_list, ex_list, d_list)

        #  num_blocks = 20
        ks_ir = [self.ks_map[x] for x in ks_list]
        ex_ir = [self.ex_map[x] for x in ex_list]
        r_ir = [(r[0] - 112) // 16]
        return ks_ir + ex_ir + d_list + r_ir


    def ir2feats(self, ir_list):
        len_ = 21 if 'proxyless' in self.network_type else 20
        ks_ir = ir_list[: len_]
        ex_ir = ir_list[len_: 2 * len_]
        d_ir = ir_list[2 * len_: -1]
        r_ir = ir_list[-1]
        start = 0
        end = 4
        for d in d_ir[:-1]:
            for j in range(start + d, end):
                ks_ir[j] = -1
                ex_ir[j] = -1
            start += 4
            end += 4
#
        ks_onehot = [0 for _ in range(len_ * 3)]
        ex_onehot = [0 for _ in range(len_ * 3)]
        r_onehot = [0 for _ in range(8)]
        for i in range(len_):
            start = i * 3
            if ks_ir[i] != -1:
                ks_onehot[start + ks_ir[i]] = 1
                ex_onehot[start + ex_ir[i]] = 1

        r_onehot[r_ir] = 1
        return ks_onehot + ex_onehot + r_onehot

    def feats2ir(self, enc_list):
        len_ = 63 if 'proxyless' in self.network_type else 60
        ks_onehot = enc_list[:len_]
        ex_onehot = enc_list[len_:2 * len_]
        r_onehot = enc_list[2 * len_:]


        ks_list = []
        ex_list = []
        d_list = []
        for i in range(0, len_, 3):
            stage_ks = np.array(ks_onehot[i:i + 3])
            stage_ex = np.array(ex_onehot[i:i + 3])
            ks_idx = np.argwhere(stage_ks == 1)
            ex_idx = np.argwhere(stage_ex == 1)

            if ks_idx.shape[0] == 0:
                ks_list.append(-1)
                ex_list.append(-1)
            else:
                ks_list.append(ks_idx[0][0])
                ex_list.append(ex_idx[0][0])

        stages = 5 if 'proxyless' in self.network_type else 6
        for stage in range(stages):
            is_skip = False
            for d in range(4):
                if ks_list[stage * 4 + d] == -1:
                    d_list.append(d)
                    is_skip = True
                    break
            if not is_skip:
                d_list.append(4)

        if 'proxyless' in self.network_type:
            d_list.append(1)

        r = np.argwhere(r_onehot)[0][0]
        return ks_list + ex_list + d_list + [r]

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
    est.save_model(args.save_path)
    est = BaseEstimator()
    est.load_model(args.save_path)
    print(est.predict(X_test[0].reshape(1, -1)), y_test[0])
    from sklearn.metrics import mean_squared_error
    print(mean_squared_error(est.predict(X_test), y_test, squared=False))


if __name__ == '__main__':

    gen_latency_model()
    '''
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
    from sklearn.metrics import mean_squared_error
    print(mean_squared_error(est.predict(X_test), y_test, squared=False))
    '''
