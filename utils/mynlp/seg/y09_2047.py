# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import sys
import gzip
import marshal
from math import log

from ..utils import frequency


class CharacterBasedGenerativeModel(object):

    def __init__(self):
        self.l1 = 0.0
        self.l2 = 0.0
        self.l3 = 0.0
        self.status = ('b', 'm', 'e', 's')
        self.uni = frequency.NormalProb()
        self.bi = frequency.NormalProb()
        self.tri = frequency.NormalProb()

    def save(self, fname, iszip=True):
        d = {}
        for k, v in self.__dict__.items():
            if hasattr(v, '__dict__'):
                d[k] = v.__dict__
            else:
                d[k] = v
        if sys.version_info[0] == 3:
            fname = fname + '.3'
        if not iszip:
            marshal.dump(d, open(fname, 'wb'))
        else:
            f = gzip.open(fname, 'wb')
            f.write(marshal.dumps(d))
            f.close()

    def load(self, fname, iszip=True):
        if sys.version_info[0] == 3:
            fname = fname + '.3'
        if not iszip:
            d = marshal.load(open(fname, 'rb'))
        else:
            try:
                f = gzip.open(fname, 'rb')
                d = marshal.loads(f.read())
            except IOError:
                f = open(fname, 'rb')
                d = marshal.loads(f.read())
            f.close()
        for k, v in d.items():
            if hasattr(self.__dict__[k], '__dict__'):
                self.__dict__[k].__dict__ = v
            else:
                self.__dict__[k] = v

    def div(self, v1, v2):
        if v2 == 0:
            return 0
        return float(v1)/v2

    def train(self, data):
        for sentence in data:
            now = [('', 'BOS'), ('', 'BOS')]
            self.bi.add((('', 'BOS'), ('', 'BOS')), 1)
            self.uni.add(('', 'BOS'), 2)
            for word, tag in sentence:
                now.append((word, tag))
                self.uni.add((word, tag), 1)
                self.bi.add(tuple(now[1:]), 1)
                self.tri.add(tuple(now), 1)
                now.pop(0)
        tl1 = 0.0
        tl2 = 0.0
        tl3 = 0.0
        samples = sorted(self.tri.samples(), key=lambda x: self.tri.get(x)[1])
        for now in samples:
            c3 = self.div(self.tri.get(now)[1]-1, self.bi.get(now[:2])[1]-1)
            c2 = self.div(self.bi.get(now[1:])[1]-1, self.uni.get(now[1])[1]-1)
            c1 = self.div(self.uni.get(now[2])[1]-1, self.uni.getsum()-1)
            if c3 >= c1 and c3 >= c2:
                tl3 += self.tri.get(now)[1]
            elif c2 >= c1 and c2 >= c3:
                tl2 += self.tri.get(now)[1]
            elif c1 >= c2 and c1 >= c3:
                tl1 += self.tri.get(now)[1]
        self.l1 = self.div(tl1, tl1+tl2+tl3)
        self.l2 = self.div(tl2, tl1+tl2+tl3)
        self.l3 = self.div(tl3, tl1+tl2+tl3)

    def log_prob(self, s1, s2, s3):
        uni = self.l1*self.uni.freq(s3)
        bi = self.div(self.l2*self.bi.get((s2, s3))[1], self.uni.get(s2)[1])
        tri = self.div(self.l3*self.tri.get((s1, s2, s3))[1],
                       self.bi.get((s1, s2))[1])
        if uni+bi+tri == 0:
            return float('-inf')
        return log(uni+bi+tri)

    def tag(self, data):
        now = [((('', 'BOS'), ('', 'BOS')), 0.0, [])]
        for w in data:
            stage = {}
            not_found = True
            for s in self.status:
                if self.uni.freq((w, s)) != 0:
                    not_found = False
                    break
            if not_found:
                for s in self.status:
                    for pre in now:
                        stage[(pre[0][1], (w, s))] = (pre[1], pre[2]+[s])
                now = list(map(lambda x: (x[0], x[1][0], x[1][1]),
                               stage.items()))
                continue
            for s in self.status:
                for pre in now:
                    p = pre[1]+self.log_prob(pre[0][0], pre[0][1], (w, s))
                    if (not (pre[0][1],
                             (w, s)) in stage) or p > stage[(pre[0][1],
                                                            (w, s))][0]:
                        stage[(pre[0][1], (w, s))] = (p, pre[2]+[s])
            now = list(map(lambda x: (x[0], x[1][0], x[1][1]), stage.items()))
        return zip(data, max(now, key=lambda x: x[1])[2])
