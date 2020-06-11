#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import time
import os
import torch
import argparse
import random

from imagenet_codebase.data_providers.imagenet import ImagenetDataProvider
from imagenet_codebase.run_manager import ImagenetRunConfig, RunManager
from model_zoo import ofa_net

from imagenet_codebase.utils.pytorch_utils import measure_net_latency, get_net_info

parser = argparse.ArgumentParser()
parser.add_argument('-p',
                    '--path',
                    help='The path of imagenet',
                    type=str,
                    default='/dataset/imagenet')
parser.add_argument('-g',
                    '--gpu',
                    help='The gpu(s) to use',
                    type=str,
                    default='all')
parser.add_argument('-b',
                    '--batch-size',
                    help='The batch on every device for validation',
                    type=int,
                    default=100)
parser.add_argument('-j',
                    '--workers',
                    help='Number of workers',
                    type=int,
                    default=20)
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

parser.add_argument('--num_sample', type=int, default=200)
parser.add_argument('-o', '--output', type=str, default='./.tmp/samples')
parser.add_argument('--gpu_bs', type=int, default=32)
parser.add_argument('--dy_img_size', action="store_true")
parser.add_argument('--interval', type=int, default=50)

args = parser.parse_args()
if args.gpu == 'all':
    device_list = range(torch.cuda.device_count())
    args.gpu = ','.join(str(_) for _ in device_list)
else:
    device_list = [int(_) for _ in args.gpu.split(',')]
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.batch_size = args.batch_size * max(len(device_list), 1)
ImagenetDataProvider.DEFAULT_PATH = args.path

if not os.path.exists(args.output):
    os.makedirs(args.output)

ofa_network = ofa_net(args.net, pretrained=True)
run_config = ImagenetRunConfig(test_batch_size=args.batch_size,
                               n_worker=args.workers)

sample_list = []

for iter_ in range(args.num_sample):
    d = ofa_network.sample_active_subnet()
    subnet = ofa_network.get_active_subnet(preserve_weight=True)
    run_manager = RunManager('.tmp/eval_subnet',
                             subnet,
                             run_config,
                             init=False)
    img_size = 224 if not args.dy_img_size else random.choice(
        [128, 160, 192, 224])
    #  run_config.data_provider.assign_active_img_size(224)
    run_config.data_provider.assign_active_img_size(img_size)
    run_manager.reset_running_statistics(net=subnet)
    loss, top1, top5 = run_manager.validate(net=subnet)
    net_info = {
        'dim':d,
        'avg_meter': (loss, top1, top5),
        'img_size': img_size
    }
    sample_list.append(net_info)
    del subnet
    if iter_ % args.interval == 0:
        output_path = os.path.join(
            args.output, '{}.json'.format(
                time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())))

        with open(output_path, 'w') as f:
            f.write(json.dumps(sample_list))
        sample_list = []
