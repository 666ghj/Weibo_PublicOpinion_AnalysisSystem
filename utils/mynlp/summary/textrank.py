# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from ..sim.bm25 import BM25


class TextRank(object):

    def __init__(self, docs):
        self.docs = docs
        self.bm25 = BM25(docs)
        self.D = len(docs)
        self.d = 0.85
        self.weight = []
        self.weight_sum = []
        self.vertex = []
        self.max_iter = 200
        self.min_diff = 0.001
        self.top = []

    def solve(self):
        for cnt, doc in enumerate(self.docs):
            scores = self.bm25.simall(doc)
            self.weight.append(scores)
            self.weight_sum.append(sum(scores)-scores[cnt])
            self.vertex.append(1.0)
        for _ in range(self.max_iter):
            m = []
            max_diff = 0
            for i in range(self.D):
                m.append(1-self.d)
                for j in range(self.D):
                    if j == i or self.weight_sum[j] == 0:
                        continue
                    m[-1] += (self.d*self.weight[j][i]
                              / self.weight_sum[j]*self.vertex[j])
                if abs(m[-1] - self.vertex[i]) > max_diff:
                    max_diff = abs(m[-1] - self.vertex[i])
            self.vertex = m
            if max_diff <= self.min_diff:
                break
        self.top = list(enumerate(self.vertex))
        self.top = sorted(self.top, key=lambda x: x[1], reverse=True)

    def top_index(self, limit):
        return list(map(lambda x: x[0], self.top))[:limit]

    def top(self, limit):
        return list(map(lambda x: self.docs[x[0]], self.top))


class KeywordTextRank(object):

    def __init__(self, docs):
        self.docs = docs
        self.words = {}
        self.vertex = {}
        self.d = 0.85
        self.max_iter = 200
        self.min_diff = 0.001
        self.top = []

    def solve(self):
        for doc in self.docs:
            que = []
            for word in doc:
                if word not in self.words:
                    self.words[word] = set()
                    self.vertex[word] = 1.0
                que.append(word)
                if len(que) > 5:
                    que.pop(0)
                for w1 in que:
                    for w2 in que:
                        if w1 == w2:
                            continue
                        self.words[w1].add(w2)
                        self.words[w2].add(w1)
        for _ in range(self.max_iter):
            m = {}
            max_diff = 0
            tmp = filter(lambda x: len(self.words[x[0]]) > 0,
                         self.vertex.items())
            tmp = sorted(tmp, key=lambda x: x[1] / len(self.words[x[0]]))
            for k, v in tmp:
                for j in self.words[k]:
                    if k == j:
                        continue
                    if j not in m:
                        m[j] = 1 - self.d
                    m[j] += (self.d / len(self.words[k]) * self.vertex[k])
            for k in self.vertex:
                if k in m and k in self.vertex:
                    if abs(m[k] - self.vertex[k]) > max_diff:
                        max_diff = abs(m[k] - self.vertex[k])
            self.vertex = m
            if max_diff <= self.min_diff:
                break
        self.top = list(self.vertex.items())
        self.top = sorted(self.top, key=lambda x: x[1], reverse=True)

    def top_index(self, limit):
        return list(map(lambda x: x[0], self.top))[:limit]

    def top(self, limit):
        return list(map(lambda x: self.docs[x[0]], self.top))
