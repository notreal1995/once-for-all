#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

from abc import ABCMeta, abstractmethod

import numpy as np
import xgboost as xgb

from encoder import Encoder


class BaseEstimator(metaclass=ABCMeta):
    @abstractmethod
    def load_model(self, path):
        pass

    @abstractmethod
    def save_model(self, path):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X, fmt='raw'):
        pass


class LatencyEstimator(BaseEstimator):
    def __init__(self, network_type='proxyless'):
        self._network_type = network_type
        self._model = xgb.XGBRegressor()
        self._encoder = Encoder(network_type=network_type)

    def load_model(self, path):
        self._model.load_model(path)

    def save_model(self, path):
        self._model.save_model(path)

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict(self, X, fmt='raw'):
        FUNC = {'raw': self._predict_raw, 
                'enc': self._predict_enc}
        return FUNC[fmt](X)

    def _predict_raw(self, X):
        return self._model.predict(X)

    def _predict_enc(self, X):
        enc = []
        for x in X:
            enc.append((self._encoder.enc2feats(x)))
        enc = np.array(enc)
        return self._predict_raw(enc)


class AccuarcyEstimator(BaseEstimator):
    def __init__(self, network_type='proxyless'):
        self._network_type = network_type
        self._model = xgb.XGBRegressor()
        self._encoder = Encoder(network_type=network_type)

    def load_model(self, path):
        self._model.load_model(path)

    def save_model(self, path):
        self._model.save_model(path)

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict(self, X, fmt='raw'):
        FUNC = {'raw': self._predict_raw, 
                'enc': self._predict_enc}
        return FUNC[fmt](X)

    def _predict_raw(self, X):
        return 100.0 - self._model.predict(X)

    def _predict_enc(self, X):
        enc = []
        for x in X:
            enc.append((self._encoder.enc2feats(x)))
        enc = np.array(enc)
        return self._predict_raw(enc)




if __name__ == '__main__':
    est = LatencyEstimator()
    aest = AccuarcyEstimator()
    est.load_model(
        '/home/notreal1995/Master-Thesis/curr/once-for-all/.tmp/lmodel/rtx-2080ti_batch_size_64.lmodel'
    )
    aest.load_model(
        '/home/notreal1995/Master-Thesis/curr/once-for-all/.tmp/lmodel/acc.lmodel'
    )
    from base_evolution import ArchManager
    arch = ArchManager(network_type='proxyless')
    sample = arch.random_sample()
    sample['d'][-1] = 2
    print(sample)
    enc = est._encoder.spec2enc(**sample)
    print(enc)
    feats = est._encoder.enc2feats(enc)
    S = np.array([enc, enc])
    print(est.predict(S, fmt='enc'))
    print(aest.predict(S, fmt='enc'))

    pass
