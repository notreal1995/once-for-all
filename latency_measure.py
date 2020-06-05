#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random

import torch
import tqdm

from imagenet_codebase.networks.proxyless_nets import ProxylessNASNets
from imagenet_codebase.utils.pytorch_utils import get_net_info, measure_net_latency

parser = argparse.ArgumentParser()
parser.add_argument('-b',
                    '--batch_size',
                    type=int,
                    help="Inference batch size",
                    default=1)
parser.add_argument('--sample-dir', type=str, help="path of sample set folder")
parser.add_argument('-n',
                    '--num',
                    type=int,
                    help="number for handling",
                    default=5000)
parser.add_argument('--dy-img', action='store_true')
parser.add_argument('-g', '--gpu', type=str, help="gpu to use")
args = parser.parse_args()


def dump_cfg(path, content):
    with open(path, 'w') as f:
        f.write(json.dumps(content))

img_size_list = [224] if not args.dy_img else [128, 160, 192, 224]

for i in tqdm.trange(args.num):
    curr_path = os.path.join(args.sample_dir, '{}'.format(i))
    cfg = json.load(open(os.path.join(curr_path, 'cfg.json'), 'r'))
    dim = json.load(open(os.path.join(curr_path, 'dim.json'), 'r'))
    net = ProxylessNASNets.build_from_config(cfg).cuda()
    measure_list = []
    for curr_size in img_size_list:
        net_info = get_net_info(net,
                                input_shape=(3, curr_size, curr_size), 
                                measure_latency='gpu{}'.format(args.batch_size),
                                print_info=False)

        params = net_info['params']
        flops = net_info['flops']
        latency = net_info['gpu{} latency'.format(args.batch_size)]['val']

        measure_list.append({
            'img_size': curr_size,
            'params': params,
            'flops': flops,
            'lat': latency
        })
    dump_cfg(os.path.join(curr_path, 'metrics_{}.json'.format(args.batch_size)), measure_list)

