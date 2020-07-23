#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import os
import random

import numpy as np

from tqdm import tqdm

from utils import BaseEstimator, OneHotEncoder


class ArchManager:
    def __init__(self, network_type):
        self.network_type = network_type
        if 'proxyless' in self.network_type:
            self.num_blocks = 21
        else:
            self.num_blocks = 20
        self.num_stages = 5
        self.kernel_sizes = [3, 5, 7]
        self.expand_ratios = [3, 4, 6]
        self.depths = [2, 3, 4]
        self.resolutions = [128, 160, 192, 224]

    def random_sample(self):
        sample = {}
        d = []
        e = []
        ks = []

        for i in range(self.num_stages):
            d.append(random.choice(self.depths))
        if 'proxyless' in self.network_type:
            d.append(1)

        for i in range(self.num_blocks):
            e.append(random.choice(self.expand_ratios))
            ks.append(random.choice(self.kernel_sizes))

        sample = {
            'wid': None,
            'ks': ks,
            'e': e,
            'd': d,
            'r': [random.choice(self.resolutions)]
        }

        return sample

    def random_resample(self, sample, i):
        assert i >= 0 and i < self.num_blocks
        sample['ks'][i] = random.choice(self.kernel_sizes)
        sample['e'][i] = random.choice(self.expand_ratios)

    def random_resample_depth(self, sample, i):
        assert i >= 0 and i < self.num_stages
        sample['d'][i] = random.choice(self.depths)

    def random_resample_resolution(self, sample):
        sample['r'][0] = random.choice(self.resolutions)


class EvolutionFinder(object):

    valid_constraint_range = {'rtx-2080ti': [3, 25]}

    def __init__(self, constraint_type, efficiency_constraint,
                 efficiency_predictor, accuracy_predictor, **kwargs):
        self.constraint_type = constraint_type
        self.efficiency_constraint = efficiency_constraint
        self.network_type = kwargs.get('proxyless', 'proxyless')
        # Latency Predictor
        self.efficiency_predictor = efficiency_predictor
        # One-Hot Encoder
        self.one_hot_encoder = OneHotEncoder(network_type=self.network_type)
        self.accuracy_predictor = accuracy_predictor
        self.arch_manager = ArchManager(network_type=self.network_type)
        self.num_blocks = self.arch_manager.num_blocks
        self.num_stages = self.arch_manager.num_stages

        self.mutate_prob = kwargs.get('mutate_prob', 0.2)
        self.population_size = kwargs.get('population_size', 100)
        self.max_time_budget = kwargs.get('max_time_budget', 500)
        self.parent_ratio = kwargs.get('parent_ratio', 0.5)
        self.mutation_ratio = kwargs.get('mutation_ratio', 0.5)

    def random_sample(self):
        constraint = self.efficiency_constraint
        while True:
            sample = self.arch_manager.random_sample()
            #  enc = self.one_hot_encoder.spec2feats(**sample)
            enc = self.one_hot_encoder.spec2feats(ks_list=sample['ks'],
                                                  ex_list=sample['e'],
                                                  d_list=sample['d'],
                                                  r=sample['r'][0])
            efficiency = self.efficiency_predictor.predict(
                np.array(enc).reshape(1, -1))
            if efficiency <= constraint:
                return sample, efficiency

    def mutate_sample(self, sample):
        constraint = self.efficiency_constraint
        while True:
            new_sample = copy.deepcopy(sample)

            if random.random() < self.mutate_prob:
                self.arch_manager.random_resample_resolution(new_sample)

            for i in range(self.num_blocks):
                if random.random() < self.mutate_prob:
                    self.arch_manager.random_resample(new_sample, i)

            for i in range(self.num_stages):
                if random.random() < self.mutate_prob:
                    self.arch_manager.random_resample_depth(new_sample, i)

            enc = self.one_hot_encoder.spec2feats(ks_list=new_sample['ks'],
                                                  ex_list=new_sample['e'],
                                                  d_list=new_sample['d'],
                                                  r=new_sample['r'][0])
            efficiency = self.efficiency_predictor.predict(
                np.array(enc).reshape(1, -1))
            if efficiency <= constraint:
                return new_sample, efficiency

    def crossover_sample(self, sample1, sample2):
        constraint = self.efficiency_constraint
        while True:
            new_sample = copy.deepcopy(sample1)
            for key in new_sample.keys():
                if not isinstance(new_sample[key], list):
                    continue
                for i in range(len(new_sample[key])):
                    new_sample[key][i] = random.choice(
                        [sample1[key][i], sample2[key][i]])

            enc = self.one_hot_encoder.spec2feats(ks_list=new_sample['ks'],
                                                  ex_list=new_sample['e'],
                                                  d_list=new_sample['d'],
                                                  r=new_sample['r'][0])
            efficiency = self.efficiency_predictor.predict(
                np.array(enc).reshape(1, -1))
            if efficiency <= constraint:
                return new_sample, efficiency

    def run(self, verbose=False):

        max_time_budget = self.max_time_budget
        population_size = self.population_size
        mutation_numbers = int(round(self.mutation_ratio * population_size))
        parents_size = int(round(self.parent_ratio * population_size))
        constraint = self.efficiency_constraint

        best_valids = [-100]
        population = []
        child_pool = []
        efficiency_pool = []
        best_info = None
        if verbose:
            print('Generate random population...')
        for _ in range(population_size):
            sample, efficiency = self.random_sample()
            child_pool.append(sample)
            efficiency_pool.append(efficiency)

        enc_child_pool = []
        for child in child_pool:
            enc = self.one_hot_encoder.spec2feats(ks_list=child['ks'],
                                                  ex_list=child['e'],
                                                  d_list=child['d'],
                                                  r=child['r'][0])

            enc_child_pool.append(enc)
        accs = self.accuracy_predictor.predict(np.array(enc_child_pool))
        #  print(accs)
        for i in range(mutation_numbers):
            population.append((accs[i], child_pool[i], efficiency_pool[i]))

        #  for i in range(mutation_numbers):
        #  print(population[i])

        for iter_ in tqdm(range(max_time_budget),
                          desc="Searching with {} constraint {}".format(
                              self.constraint_type,
                              self.efficiency_constraint)):
            parents = sorted(population, key=lambda x: x[0])[::-1][:parents_size]
            acc = parents[0][0]

            if acc > best_valids[-1]:
                best_valids.append(acc)
                best_info = parents[0]
            else:
                best_valids.append(best_valids[-1])

            population = parents
            child_pool = []
            efficiency_pool = []

            for i in range(mutation_numbers):
                par_sample = population[np.random.randint(parents_size)][1]
                new_sample, efficiency = self.mutate_sample(par_sample)
                child_pool.append(new_sample)
                efficiency_pool.append(efficiency)

            for i in range(population_size - mutation_numbers):
                par_sample1 = population[np.random.randint(parents_size)][1]
                par_sample2 = population[np.random.randint(parents_size)][1]
                new_sample, efficiency = self.crossover_sample(par_sample1, par_sample2)
                child_pool.append(new_sample)
                efficiency_pool.append(efficiency)

            enc_child_pool = []
            for child in child_pool:
                enc = self.one_hot_encoder.spec2feats(ks_list=child['ks'],
                                                      ex_list=child['e'],
                                                      d_list=child['d'],
                                                      r=child['r'][0])

                enc_child_pool.append(enc)
            accs = self.accuracy_predictor.predict(np.array(enc_child_pool))
            for i in range(population_size):
                population.append((accs[i], child_pool[i], efficiency_pool[i]))

        return best_valids, best_info

if __name__ == '__main__':
    #  archManager = ArchManager(network_type='proxyless')
    #  print(archManager.random_sample())
    est = BaseEstimator()
    acc_est = BaseEstimator()
    est.load_model(
        '/home/notreal1995/Master-Thesis/curr/once-for-all/.tmp/lmodel/rtx-2080ti_batch_size_64.lmodel'
    )
    acc_est.load_model(
        '/home/notreal1995/Master-Thesis/curr/once-for-all/.tmp/lmodel/acc-new.lmodel'
    )
    evolution_finder = EvolutionFinder(constraint_type='rtx-2080ti',
                                       efficiency_constraint=5,
                                       efficiency_predictor=est,
                                       accuracy_predictor=acc_est)

    best_valids, best_info = evolution_finder.run(verbose=True)
    print(best_valids)
    print(best_info)
    evolution_finder = EvolutionFinder(constraint_type='rtx-2080ti',
                                       efficiency_constraint=15,
                                       efficiency_predictor=est,
                                       accuracy_predictor=acc_est)

    best_valids, best_info = evolution_finder.run(verbose=True)
    print(best_valids)
    print(best_info)
    evolution_finder = EvolutionFinder(constraint_type='rtx-2080ti',
                                       efficiency_constraint=20,
                                       efficiency_predictor=est,
                                       accuracy_predictor=acc_est)

    best_valids, best_info = evolution_finder.run(verbose=True)
    print(best_valids)
    print(best_info)
