# -*- coding: utf-8 -*-
from __future__ import unicode_literals


class SimpleMerge(object):

    def __init__(self, doc, words):
        self.doc = doc
        self.words = words

    def merge(self):
        trans = {}
        for w in self.words:
            trans[w] = ''
        for w1 in self.words:
            cw = 0
            lw = len(w1)
            for i in range(len(self.doc)-lw+1):
                if w1 == self.doc[i: i+lw]:
                    cw += 1
            for w2 in self.words:
                cnt = 0
                l2 = len(w1)+len(w2)
                for i in range(len(self.doc)-l2+1):
                    if w1+w2 == self.doc[i: i+l2]:
                        cnt += 1
                if cw < cnt*2:
                    trans[w1] = w2
                    break
        ret = []
        for w in self.words:
            if w not in trans:
                continue
            s = ''
            now = trans[w]
            while now:
                s += now
                if now not in trans:
                    break
                tmp = trans[now]
                del trans[now]
                now = tmp
            trans[w] = s
        for w in self.words:
            if w in trans:
                ret.append(w+trans[w])
        return ret
