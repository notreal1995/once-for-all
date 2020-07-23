#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import tqdm
import torch
import torch.nn as nn

from ofa.model_zoo import ofa_net

from mywork.data_providers.cifar import CIFAR10DataProvider, CIFAR100DataProvider
from ofa.imagenet_codebase.modules.layers import LinearLayer


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--epochs', type=int, default=600, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight-decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--val_interval', type=int, default=5, help='validate and save frequency')

args = parser.parse_args()

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(args, epoch, train_data, device, model, criterion, optimizer, scheduler):
    model.train()
    train_loss = 0.0
    top1 = AvgrageMeter()
    train_data = tqdm.tqdm(train_data)
    train_data.set_description('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, args.epochs, 'lr:', scheduler.get_last_lr()[0]))
    for step, (inputs, targets) in enumerate(train_data):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # if args.dataset == 'cifar10':
        loss.backward()
        # elif args.dataset == 'imagenet':
        #     with amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        optimizer.step()
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        top1.update(prec1.item(), n)
        train_loss += loss.item()
        postfix = {'train_loss': '%.6f' % (train_loss / (step + 1)), 'train_acc': '%.6f' % top1.avg}
        train_data.set_postfix(log=postfix)

def validate(args, epoch, val_data, device, model, criterion):
    model.eval()
    val_loss = 0.0
    val_top1 =AvgrageMeter()
    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)
        print('[Val_Accuracy epoch:%d] val_loss:%f, val_acc:%f'
              % (epoch + 1, val_loss / (step + 1), val_top1.avg))
        return val_top1.avg

if torch.cuda.is_available():
    print('Train on GPU!')
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

ofa_network = ofa_net('ofa_proxyless_d234_e346_k357_w1.3', pretrained=True)
ofa_network.sample_active_subnet()
model = ofa_network.get_active_subnet(preserve_weight=True)

data_provider = CIFAR100DataProvider(save_path='/tmp')

#  model.classifier = LinearLayer(in_features=model.classifier.in_features, out_features=10)
model.classifier = LinearLayer(in_features=model.classifier.in_features, out_features=100)

model = model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - (epoch / args.epochs))


for epoch in range(args.epochs):
    train(args, epoch, data_provider.train, device, model, criterion, optimizer, scheduler)
    scheduler.step()
    if (epoch + 1) % args.val_interval == 0:
            validate(args, epoch, data_provider.test, device, model, criterion)
