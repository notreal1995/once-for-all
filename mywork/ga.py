#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import random

import numpy as np

from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.model.sampling import Sampling
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.operators.crossover.util import crossover_mask
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.algorithms.nsga2 import NSGA2

from utils import BaseEstimator, OneHotEncoder
from base_evolution import ArchManager
from encoder import Encoder
from estimator import AccuarcyEstimator, LatencyEstimator


class ProxylessNASProblemer(Problem):
    def __init__(self, n_obj, estimators, constr=None):
        self.xl = np.full((49), 0)
        self.xu = np.append(np.full((48), 2), 7)
        self.constr = constr
        self.estimators = estimators
        super().__init__(
            n_var=49,
            n_obj=n_obj,
            n_constr=0 if self.constr is None else len(self.constr),
            #  n_constr=0,
            xl=self.xl,
            xu=self.xu,
            type_var=np.int)


    def _evaluate(self, x, out, *args, **kwargs):
        devices = []
        constrs = []
        acc = self.estimators['accuracy'].predict(x, fmt='enc')
        for device, est in self.estimators['devices'].items():
            devices.append(est.predict(x, fmt='enc'))
            if self.constr.get(device):
                curr = self.constr[device]
                lo, hi = curr.get('lo'), curr.get('hi')
                if lo:
                    constrs.append(-1 *  (devices[-1] - lo))
                if hi:
                    constrs.append(devices[-1] - hi)

        #  others = [f.predict(x, fmt='enc') for f in self.estimators['others'].values()]
        #  perf = np.column_stack(
            #  [f.predict(x, fmt='enc') for f in self.estimators])
        #  out['F'] = perf
        out['F'] = np.column_stack(devices + [acc])
        out['G'] = np.column_stack(constrs)


def main():
    acc_est = AccuarcyEstimator()
    lat_est = LatencyEstimator()
    lat_est.load_model(
        #  '/home/notreal1995/Master-Thesis/curr/once-for-all/.tmp/lmodel/rtx-2080ti_batch_size_64.lmodel'
        '/home/notreal1995/Master-Thesis/curr/once-for-all/.tmp/lmodel/jetson-nano_batch_size_1_new.lmodel'
        #  '/home/notreal1995/Master-Thesis/curr/once-for-all/.tmp/lmodel/xavier_batch_size_32.lmodel'
    )
    acc_est.load_model(
        '/home/notreal1995/Master-Thesis/curr/once-for-all/.tmp/lmodel/acc.lmodel'
    )
    ests = {
        'accuracy': acc_est,
        'devices': {
            'RTX-2080ti': lat_est
        }
    }
    constr = {
        'RTX-2080ti': {
            'lo': 10
        }
    }
    problemer = ProxylessNASProblemer(n_obj=2, estimators=ests, constr=constr)

    algorithm = NSGA2(
        pop_size=100,
        n_offsprings=50,
        sampling=get_sampling("int_random"),
        #  crossover=get_crossover("int_ux"),
        crossover=get_crossover('int_sbx', prob=0.6, eta=1),
        mutation=get_mutation('int_pm', eta=1),
        eliminate_duplicates=True)
    termination = get_termination("n_gen", 500)

    #
    #
    res = minimize(
        problemer,
        algorithm,
        termination,
        #  seed=87,
        save_history=True,
        verbose=True)
    F = res.pop.get("F")
    X = res.pop.get("X")
    for i in range(F.shape[0]):
        x = X[i]
        print(lat_est._encoder.enc2spec(x))
        print(F[i])

    #  from pymoo.visualization.scatter import Scatter
    #  Scatter().add(res.F).save('res.png')
    import matplotlib.pyplot as plt
    x, y = F[:, 0], 100.0 - F[:, 1]
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=4)
    ax.set(title='Nvidia Jetson-Nano',
           ylabel='Accuarcy (%)',
           xlabel='Latency (ms) with batch size 1')
    plt.savefig('nano-new.png')


if __name__ == '__main__':
    main()
