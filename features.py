import glob
import re

from os import path

import numpy as np

from utils_common import dictlist_append, raw_filename
from utils_text import get_window, numWords



"""
We use subsequent format of data:
each article is presented by two files: xxx.txt and xxx.key where xxx is some filename.
The former file has in it an article divided into parts like this:
--T
title
--A
abstract
--B
body
--R
references

The latter contains marked keywords. The default marking is following:
-T
task keywords
-M
method keywords
-O
other keywords
"""
def get_text(filename):
    """
    retrieve text of a given article
    :param filename: article filename
    :return: a map between marks and parts of an article: {'Mark1':'text of a part1', 'Mark2':'text of a part2'}
    """

    with open(filename, 'r') as f:
        lines = map(lambda x: x.rstrip(), f.readlines())
        it = iter(lines)
        n = next(it, None)

        mp = {}

        while n is not None:
            while n is not None and n.startswith('--'):
                chlab = n[2:]
                if chlab not in mp:
                    mp[chlab] = ''
                currLabel = chlab
                n = next(it, None)

            while n is not None and not n.startswith('--'):
                if n != '':
                    mp[currLabel]+=n
                n = next(it, None)
    return mp


def text_features(filename, keyword):
    """
    build features considering a pair keyword-article
    :param filename: article filename
    :param keyword: keyword
    :return: tuple of numerical features
    """
    text = get_text(filename)

    tcont = text['T'].count(keyword) > 0
    acont = text['A'].count(keyword) > 0
    bcont = text['B'].count(keyword) > 0
    rcont = text['R'].count(keyword) > 0
    trcont = text['TR'].count(keyword) > 0

    if bcont > 0:
        pos = text['B'].index(keyword)
        firstoc = float(len(re.findall('\ +', text['B'][:pos])))/len(re.findall('\ +', text['B']))
    else:
        firstoc = -1
    return tcont, acont, bcont, rcont, trcont, firstoc


def get_keywords(filename, labelmap):
    """
    generates labelled dictionary of keywords placed by label
    :param filename: marked .key file
    :param labelmap: map from string-labels into int-labels
    :return: dictionary as {0: [kw0, kw1], 1: [], 2:[]}
    """
    keywords = {}

    if labelmap is None:
        labelmap = {}

    with open(filename, 'r') as f:
        lines = map(lambda x: x.rstrip(), f.readlines())
        it = iter(lines)
        n = next(it, None)

        maxLabel = 0

        while n is not None:
            while n is not None and n.startswith('-'):
                chlab = n[1:]
                if chlab not in labelmap:
                    labelmap[chlab] = maxLabel
                    maxLabel += 1
                currLabel = labelmap[chlab]
                n = next(it, None)

            while n is not None and not n.startswith('-'):
                if n != '':
                    dictlist_append(keywords, currLabel, n)
                n = next(it, None)

    return keywords

def keyword_features(kw):
    """
    builds features considering only a keyword
    :param kw: keyword
    :return: tuple of numerical features
    """
    if kw.endswith('s'):
        kw = kw[:-1]

    num = numWords(kw)
    tyend = False
    words = kw.split()
    for word in words:
        if word.endswith('ty'):
            tyend = True

    hasalg = 'algorithm' in kw

    return num, tyend, hasalg

def get_data_byfile(filename, labelmap, wind_st=-5, wind_end=5):
    """
    build features and word windows for one article
    :param filename: name of the keywords and article file
    :param labelmap:
    :return:
    """
    data = []
    kwlist = []
    windows = []

    keyfile = '{}.key'.format(filename)
    txtfile = '{}.txt'.format(filename)

    kws = get_kwlist_byfile(keyfile, labelmap)
    tab = get_text(txtfile)

    text = '. '.join([tab['A'], tab['B']])

    is_part = [0]*len(kws)

    for i in range(len(kws)):
        for j in range(i+1, len(kws)):
            if kws[i][1] in kws[j][1] or kws[j][1] in kws[i][1]:
                is_part[i] = 1
                is_part[j] = 1


    for i in range(len(kws)):
        kw_row = kws[i]
        rawFname = kw_row[0]
        kw = kw_row[1]
        label = kw_row[2]
        kwlist.append((rawFname, kw))
        row = (text_features(txtfile, kw) + keyword_features(kw) + (is_part[i], label))
        data.append(row)

        window = get_window(text, kw, wind_st, wind_end, outtype='text')

        windows.append(window)

    return data, kwlist, windows


def get_kwlist_byfile(filename, labelmap):
    data = []
    keywords_mapped = get_keywords(filename, labelmap)
    for label in keywords_mapped:
        for kw in keywords_mapped[label]:
            data.append((raw_filename(filename)[1], kw, label))
    return data


def build_data(directory, labelmap=None):
    if labelmap is None:
        labelmap = {'T': 0, 'M': 1, 'O': 2}

    data = []
    kwlist = []

    windows = []
    windows_aft = []

    dir_re = path.join('{}','*.key').format(directory)
    filenames = glob.glob(dir_re)
    for filename in filenames:
        directory, file = raw_filename(filename)
        filename = path.join(directory, file)
        file_data, file_kws, window = get_data_byfile(filename, labelmap)
        data += file_data
        kwlist += file_kws

        windows += window


    return np.array(data), kwlist, windows

