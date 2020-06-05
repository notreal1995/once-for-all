#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time

from model_zoo import ofa_net

parser = argparse.ArgumentParser()
parser.add_argument('-n',
                    '--net',
                    metavar='OFANET',
                    default='ofa_proxyless_d234_e346_k357_w1.3',
                    choices=[
                        'ofa_mbv3_d234_e346_k357_w1.0',
                        'ofa_mbv3_d234_e346_k357_w1.2',
                        'ofa_proxyless_d234_e346_k357_w1.3'
                    ],
                    help='OFA networks')
parser.add_argument('-o', '--output', type=str, default='./.tmp/samples')
parser.add_argument('-g',
                    '--gpu',
                    help='The gpu(s) to use',
                    type=str,
                    default='all')
parser.add_argument('--gpu_bs', type=int, default=64)


parser.add_argument('--num', type=int, default=5000)
args = parser.parse_args()

folder_path = os.path.join(
    args.output,
    '{}_sample_{}'.format(args.net, args.num))
os.makedirs(folder_path, exist_ok=True)

ofa_network = ofa_net(args.net, pretrained=False)

def dump_cfg(path, content):
    with open(path, 'w') as f:
        f.write(json.dumps(content))

for i in range(args.num):
    sub_fold_path = os.path.join(folder_path, '{}'.format(i))
    os.makedirs(sub_fold_path, exist_ok=True)
    net_info = ofa_network.sample_active_subnet()
    cfg = ofa_network.get_active_subnet().config
    dump_cfg(os.path.join(sub_fold_path, 'dim.json'), net_info)
    dump_cfg(os.path.join(sub_fold_path, 'cfg.json'), cfg)


#  sample_list = []

#  ofa_network = ofa_net(args.net, pretrained=False)
#  for i in range(args.num):
#  net_dim = ofa_network.sample_active_subnet()
#  sample_list.append(net_dim)
