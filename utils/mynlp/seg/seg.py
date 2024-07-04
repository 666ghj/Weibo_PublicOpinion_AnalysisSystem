# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals

import codecs

from ..utils.tnt import TnT
from .y09_2047 import CharacterBasedGenerativeModel


class Seg(object):

    def __init__(self, name='other'):
        if name == 'tnt':
            self.segger = TnT()
        else:
            self.segger = CharacterBasedGenerativeModel()

    def save(self, fname, iszip=True):
        self.segger.save(fname, iszip)

    def load(self, fname, iszip=True):
        self.segger.load(fname, iszip)

    def train(self, fname):
        fr = codecs.open(fname, 'r', 'utf-8')
        data = []
        for i in fr:
            line = i.strip()
            if not line:
                continue
            tmp = map(lambda x: x.split('/'), line.split())
            data.append(tmp)
        fr.close()
        self.segger.train(data)

    def seg(self, sentence):
        ret = self.segger.tag(sentence)
        tmp = ''
        for i in ret:
            if i[1] == 'e':
                yield tmp+i[0]
                tmp = ''
            elif i[1] == 'b' or i[1] == 's':
                if tmp:
                    yield tmp
                tmp = i[0]
            else:
                tmp += i[0]
        if tmp:
            yield tmp


if __name__ == '__main__':
    seg = Seg()
    seg.train('data.txt')
    print(' '.join(seg.seg('主要是用来放置一些简单快速的中文分词和词性标注的程序')))
