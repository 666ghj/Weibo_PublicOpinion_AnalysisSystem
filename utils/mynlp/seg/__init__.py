# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
import re

from . import seg as TnTseg

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'seg.marshal')
segger = TnTseg.Seg()
segger.load(data_path, True)
re_zh = re.compile('([\u4E00-\u9FA5]+)')


def seg(sent):
    words = []
    for s in re_zh.split(sent):
        s = s.strip()
        if not s:
            continue
        if re_zh.match(s):
            words += single_seg(s)
        else:
            for word in s.split():
                word = word.strip()
                if word:
                    words.append(word)
    return words


def train(fname):
    global segger
    segger = TnTseg.Seg()
    segger.train(fname)


def save(fname, iszip=True):
    segger.save(fname, iszip)


def load(fname, iszip=True):
    segger.load(fname, iszip)


def single_seg(sent):
    return list(segger.seg(sent))
