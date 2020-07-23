#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Encoder:
    def __init__(self, network_type='proxyless'):
        self._network_type = network_type
        self.length = 21 if 'proxyless' in network_type else 20
        self._init_map()
        pass

    @staticmethod
    def construct_maps(keys):
        d = dict()
        keys = list(set(keys))
        for k in keys:
            if k not in d:
                d[k] = len(list(d.keys()))
        return d

    def _init_map(self):
        self.ks_map = self.construct_maps(keys=(3, 5, 7))
        self.ex_map = self.construct_maps(keys=(3, 4, 6))
        self.dp_map = self.construct_maps(keys=(2, 3, 4))
        self.inv_ks_map = {v: k for k, v in self.ks_map.items()}
        self.inv_ex_map = {v: k for k, v in self.ex_map.items()}
        self.inv_dp_map = {v: k for k, v in self.dp_map.items()}

    def spec2enc(self, **spec):
        ks_list = spec['ks']
        ex_list = spec['e']
        d_list = spec['d']
        r_list = spec['r']
        enc_ks = [self.ks_map[x] for x in ks_list]
        enc_ex = [self.ex_map[x] for x in ex_list]
        enc_d = [self.dp_map[x] for x in d_list]
        enc_r = [(r - 112) // 16 for r in r_list]

        return enc_ks + enc_ex + enc_d + enc_r

    def enc2spec(self, enc):
        enc_ks = enc[:self.length]
        enc_ex = enc[self.length:2 * self.length]
        enc_d = enc[2 * self.length:-1]
        enc_r = enc[-1]
        ks_list = [self.inv_ks_map[x] for x in enc_ks]
        ex_list = [self.inv_ex_map[x] for x in enc_ex]
        d_list = [self.inv_dp_map[x] for x in enc_d]
        r_list = [enc_r * 16 + 112]
        return {
            'ks': ks_list,
            'e': ex_list,
            'd': d_list,
            'r': r_list
        }

    def enc2feats(self, enc):
        enc_ks = enc[:self.length]
        enc_ex = enc[self.length:2 * self.length]
        enc_d = enc[2 * self.length:-1]
        enc_r = enc[-1]
        d_list = [self.inv_dp_map[x] for x in enc_d]

        start = 0
        end = 4
        for d in d_list[:-1]:
            for j in range(start + d, end):
                enc_ks[j] = -1
                enc_ex[j] = -1
            start += 4
            end += 4


        ks_onehot = [0 for _ in range(self.length * 3)]
        ex_onehot = [0 for _ in range(self.length * 3)]
        r_onehot = [0 for _ in range(8)]

        for i in range(self.length):
            start = i * 3
            if enc_ks[i] != -1:
                ks_onehot[start + enc_ks[i]] = 1
            if enc_ex[i] != -1:
                ex_onehot[start + enc_ex[i]] = 1

        r_onehot[enc_r] = 1

        return ks_onehot + ex_onehot + r_onehot


if __name__ == '__main__':
    enc = Encoder()
    from base_evolution import ArchManager
    arch = ArchManager(network_type='proxyless')
    sample = arch.random_sample()
    sample['d'][-1] = 4
    print(enc.spec2enc(**sample))

    pass
