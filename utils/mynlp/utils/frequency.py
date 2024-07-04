# -*- coding: utf-8 -*-

from . import good_turing

class BaseProb(object):

    def __init__(self):
        self.d = {}
        self.total = 0.0
        self.none = 0

    def exists(self, key):
        return key in self.d

    def getsum(self):
        return self.total

    def get(self, key):
        if not self.exists(key):
            return False, self.none
        return True, self.d[key]

    def freq(self, key):
        return float(self.get(key)[1])/self.total

    def samples(self):
        return self.d.keys()


class NormalProb(BaseProb):

    def add(self, key, value):
        if not self.exists(key):
            self.d[key] = 0
        self.d[key] += value
        self.total += value


class AddOneProb(BaseProb):

    def __init__(self):
        self.d = {}
        self.total = 0.0
        self.none = 1

    def add(self, key, value):
        self.total += value
        if not self.exists(key):
            self.d[key] = 1
            self.total += 1
        self.d[key] += value


class GoodTuringProb(BaseProb):

    def __init__(self):
        self.d = {}
        self.total = 0.0
        self.handled = False

    def add(self, key, value):
        if not self.exists(key):
            self.d[key] = 0
        self.d[key] += value

    def get(self, key):
        if not self.handled:
            self.handled = True
            tmp, self.d = good_turing.main(self.d)
            self.none = tmp
            self.total = sum(self.d.values())+0.0
        if not self.exists(key):
            return False, self.none
        return True, self.d[key]
