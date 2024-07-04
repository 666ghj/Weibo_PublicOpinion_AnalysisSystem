# -*- coding: utf-8 -*-

'''
Implementation of 'TnT - A Statisical Part of Speech Tagger'
'''
from __future__ import unicode_literals

import sys
import gzip
import heapq
import marshal
from math import log

from . import frequency


class TnT(object):

    def __init__(self, N=1000):
        self.N = N
        self.l1 = 0.0
        self.l2 = 0.0
        self.l3 = 0.0
        self.status = set()
        self.wd = frequency.AddOneProb()
        self.eos = frequency.AddOneProb()
        self.eosd = frequency.AddOneProb()
        self.uni = frequency.NormalProb()
        self.bi = frequency.NormalProb()
        self.tri = frequency.NormalProb()
        self.word = {}
        self.trans = {}

    def save(self, fname, iszip=True):
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, set):
                d[k] = list(v)
            elif hasattr(v, '__dict__'):
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
            if isinstance(self.__dict__[k], set):
                self.__dict__[k] = set(v)
            elif hasattr(self.__dict__[k], '__dict__'):
                self.__dict__[k].__dict__ = v
            else:
                self.__dict__[k] = v

    def tnt_div(self, v1, v2):
        if v2 == 0:
            return 0
        return float(v1)/v2

    def geteos(self, tag):
        tmp = self.eosd.get(tag)
        if not tmp[0]:
            return log(1.0/len(self.status))
        return log(self.eos.get((tag, 'EOS'))[1])-log(self.eosd.get(tag)[1])

    def train(self, data):
        for sentence in data:
            now = ['BOS', 'BOS']
            self.bi.add(('BOS', 'BOS'), 1)
            self.uni.add('BOS', 2)
            for word, tag in sentence:
                now.append(tag)
                self.status.add(tag)
                self.wd.add((tag, word), 1)
                self.eos.add(tuple(now[1:]), 1)
                self.eosd.add(tag, 1)
                self.uni.add(tag, 1)
                self.bi.add(tuple(now[1:]), 1)
                self.tri.add(tuple(now), 1)
                if word not in self.word:
                    self.word[word] = set()
                self.word[word].add(tag)
                now.pop(0)
            self.eos.add((now[-1], 'EOS'), 1)
        tl1 = 0.0
        tl2 = 0.0
        tl3 = 0.0
        for now in self.tri.samples():
            c3 = self.tnt_div(self.tri.get(now)[1]-1,
                              self.bi.get(now[:2])[1]-1)
            c2 = self.tnt_div(self.bi.get(now[1:])[1]-1,
                              self.uni.get(now[1])[1]-1)
            c1 = self.tnt_div(self.uni.get(now[2])[1]-1, self.uni.getsum()-1)
            if c3 >= c1 and c3 >= c2:
                tl3 += self.tri.get(now)[1]
            elif c2 >= c1 and c2 >= c3:
                tl2 += self.tri.get(now)[1]
            elif c1 >= c2 and c1 >= c3:
                tl1 += self.tri.get(now)[1]
        self.l1 = float(tl1)/(tl1+tl2+tl3)
        self.l2 = float(tl2)/(tl1+tl2+tl3)
        self.l3 = float(tl3)/(tl1+tl2+tl3)
        for s1 in self.status | set(('BOS',)):
            for s2 in self.status | set(('BOS',)):
                for s3 in self.status:
                    uni = self.l1*self.uni.freq(s3)
                    bi = self.tnt_div(self.l2*self.bi.get((s2, s3))[1],
                                      self.uni.get(s2)[1])
                    tri = self.tnt_div(self.l3*self.tri.get((s1, s2, s3))[1],
                                       self.bi.get((s1, s2))[1])
                    self.trans[(s1, s2, s3)] = log(uni+bi+tri)

    def tag(self, data):
        now = [(('BOS', 'BOS'), 0.0, [])]
        for w in data:
            stage = {}
            samples = self.status
            if w in self.word:
                samples = self.word[w]
            for s in samples:
                wd = log(self.wd.get((s, w))[1])-log(self.uni.get(s)[1])
                for pre in now:
                    p = pre[1]+wd+self.trans[(pre[0][0], pre[0][1], s)]
                    if (pre[0][1], s) not in stage or p > stage[(pre[0][1],
                                                                 s)][0]:
                        stage[(pre[0][1], s)] = (p, pre[2]+[s])
            stage = list(map(lambda x: (x[0], x[1][0], x[1][1]), stage.items()))
            now = heapq.nlargest(self.N, stage, key=lambda x: x[1])
        now = heapq.nlargest(1, stage, key=lambda x: x[1]+self.geteos(x[0][1]))
        return zip(data, now[0][2])
