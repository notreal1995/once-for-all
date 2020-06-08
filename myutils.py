#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import joblib

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


class LatencyEstimator(object):
    def __init__(self, model_path=None):
        self._model_path = model_path
        self._est = xgb.XGBRegressor()

    def load_pretrained(self):
        self._est = joblib.load(self._model_path)

    def save_model(self, save_path):
        joblib.dump(self._est, save_path)

    def predict(self, X):
        ret = []
        for x in X:
            ret += (list(self._est.predict(np.array(x).reshape(1, -1))))
        return ret

    def fit(self, X, y):
        self._est.fit(np.array(X), np.array(y))


class ConfigEncoder(object):
    def __init__(self, net_config, net_dict, img_size=224):
        self.cfg = net_config
        self.net_dict = net_dict
        self.curr_img_size = img_size
        self._first_conv = None
        self._feature_mix_layer = None
        self._blocks = None
        self._classifier = None
        self.init()

    def init(self):
        self.first_conv
        self.blocks
        self.feature_mix_layer
        self.classifier

    def _flatten(self, target):
        ret = []
        if isinstance(target, dict):
            for t in target.values():
                if isinstance(t, tuple):
                    ret += [*t]
                else:
                    ret.append(t)
        return ret

    def get_result(self):
        ret = []
        labeled = {'skip': 0, 'linear': 1, 'mbconv': 2, 'conv': 3}
        curr = self.first_conv
        curr['ops'] = labeled[curr['ops']]
        ret += self._flatten(curr)
        for block in self.blocks:
            curr = block
            curr['ops'] = labeled[curr['ops']]
            ret += self._flatten(curr)
        curr = self.feature_mix_layer
        curr['ops'] = labeled[curr['ops']]
        ret += self._flatten(curr)
        curr = self.classifier
        curr['ops'] = labeled[curr['ops']]
        ret += self._flatten(curr)
        return ret

    @property
    def first_conv(self):
        if self._first_conv is None:
            self._first_conv = self.parse_conv(self.cfg['first_conv'])
        return self._first_conv

    @property
    def feature_mix_layer(self):
        if self._feature_mix_layer is None:
            self._feature_mix_layer = self.parse_conv(
                self.cfg['feature_mix_layer'])
        return self._feature_mix_layer

    def parse_conv(self, target):
        if target['stride'] == 2:
            out_img = self.curr_img_size // 2
        else:
            out_img = self.curr_img_size
        ret = {
            'ops':
            'conv',
            'input':
            (target['in_channels'], self.curr_img_size, self.curr_img_size),
            'output': (target['out_channels'], out_img, out_img),
            'expand_ratio':
            1,
            'stride':
            target['stride'],
            'kernel_size':
            target['kernel_size'],
            'idskip':
            0
        }
        self.curr_img_size = out_img
        return ret

    def gen_skip(self, num, prev_channels):
        skip_list = []
        for n in range(num):
            skip_list.append({
                'ops': 'skip',
                'input': (0, 0, 0),
                'output': (0, 0, 0),
                'expand_ratio': 0,
                'stride': 0,
                'kernel_size': 0,
                'idskip': 0
            })
        return skip_list

    @property
    def blocks(self):
        if self._blocks is not None:
            return self._blocks
        self._blocks = []
        block_iter = iter(self.cfg['blocks'])
        # Stem block
        stem = self.parse_mbconv(next(block_iter))
        self._blocks.append(stem)
        prev_channels = stem['output'][0]
        runtime_depth = self.net_dict['d'][:-1] + [1]
        for idx, stage_d in enumerate(runtime_depth):
            if idx != len(runtime_depth) - 1:
                self._blocks += self.gen_skip(4 - stage_d, prev_channels)
            for d in range(stage_d):
                curr = next(block_iter)
                self._blocks.append(self.parse_mbconv(curr))

        return self._blocks

    @property
    def classifier(self):

        if self._classifier is not None:
            return self._classifier

        block = self.cfg['classifier']
        self._classifier = {
            'ops': 'linear',
            'input': (block['in_features'], 1, 1),
            'output': (block['out_features'], 1, 1),
            'expand_ratio': 1,
            'stride': 0,
            'kernel_size': 0,
            'idskip': 0
        }
        return self._classifier

    def parse_mbconv(self, target):
        main_ = target['mobile_inverted_conv']
        sc = target['shortcut']
        if main_['stride'] == 2:
            out_img = self.curr_img_size // 2
        else:
            out_img = self.curr_img_size
        ret = {
            'ops': 'mbconv',
            'input':
            (main_['in_channels'], self.curr_img_size, self.curr_img_size),
            'output': (main_['out_channels'], out_img, out_img),
            'expand_ratio': main_['expand_ratio'],
            'stride': main_['stride'],
            'kernel_size': main_['kernel_size'],
            'idskip': 1 if sc is not None else 0
        }
        self.curr_img_size = out_img
        return ret


def subdirs(path):
    for entry in os.scandir(path):
        if entry.is_dir():
            yield entry.name


def main():
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
        cfg = json.load(open(os.path.join(curr_path, 'cfg.json'), 'r'))
        dim = json.load(open(os.path.join(curr_path, 'dim.json'), 'r'))
        met_str = 'metrics_{}.json'.format(args.batch_size)
        metrics = json.load(open(os.path.join(curr_path, met_str), 'r'))
        for m in metrics:
            enc = ConfigEncoder(cfg, dim, img_size=int(m['img_size']))
            X.append(enc.get_result())
            y.append(m['lat'])

    X_train, X_test, y_train, y_test = train_test_split(np.array(X),
                                                        np.array(y),
                                                        test_size=0.2)
    est = LatencyEstimator()
    est.fit(X_train, y_train)
    est.save_model(args.save_path)
    from sklearn.metrics import mean_squared_error
    print(mean_squared_error(est.predict(X_test), y_test, squared=False))


if __name__ == '__main__':
    main()

#  if __name__ == '__main__':
#  parser = argparse.ArgumentParser()
#  from elastic_nn.networks.ofa_proxyless import OFAProxylessNASNets
#  net = OFAProxylessNASNets(dropout_rate=0,
#  width_mult_list=1.3,
#  ks_list=[3, 5, 7],
#  expand_ratio_list=[3, 4, 6],
#  depth_list=[2, 3, 4])
#  with open('/home/notreal1995/2020_05_26_12_51_44_xavier_gpu32_lat.json',
#  'r') as f:
#  with open('./.tmp/samples/2020_05_18_11_03_49_nano_gpu1_lat.json', 'r') as f:
#  sample_list = json.loads(f.read())
#
#  encoded_list = []
#
#  for i in range(len(sample_list)):
#
#  curr = sample_list[i]
#  net.set_active_subnet(curr['wid'], curr['ks'], curr['e'], curr['d'])
#  subnet = net.get_active_subnet()
#  enc = ConfigEncoder(subnet.config, curr)
#  encoded_list.append((enc.get_result(), curr['net_info']['lat']))

#  X, y = zip(*encoded_list)
#  X_train, X_test, y_train, y_test = train_test_split(np.array(X),
#  np.array(y),
#  test_size=0.2)
#  est = LatencyEstimator()
#  est.fit(X_train, y_train)
#  est.save_model('./.tmp/lmodel/jetson-xavier_batch_size_32.lmodel')
#  est.save_model('./.tmp/lmodel/jetson-nano_batch_size_1.lmodel')
#  from sklearn.metrics import mean_squared_error
#  print(mean_squared_error(est.predict(X_test), y_test, squared=False))
