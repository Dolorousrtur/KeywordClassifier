import glob
import re

import os
from os import path

import numpy as np

from utils_common import dictlist_append, raw_filename
from text_processing import get_window, wordcount

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

class KwDataset:
    def __init__(self, directory, labelmap):

        if directory.endswith(os.sep):
            directory = directory[:-1]

        self.directory = directory
        self.labelmap = labelmap
        self.build_data()


    def get_data(self):
        return self.data[:, :-1]

    def get_labels(self):
        return self.data[:, -1]

    def get_keyword_list(self):
        return self.keywords

    def build_windows(self, window_st=-5, window_end=5, outtype='list'):
        windows = []

        curr_file = ""
        text = ""

        if self.directory != "":
            txtfn = path.join('{}','{}.txt')
        else:
            txtfn = '{}{}.txt'

        for kw in self.keywords:
            filename = kw[0]
            keyword =  kw[1]
            if filename != curr_file:
                tab = get_text(txtfn.format(self.directory, filename))
                text = ' '.join((tab['A'], tab['B']))
            window = get_window(text, keyword, window_st, window_end,  outtype=outtype)
            windows.append(window)

        return windows


    def build_data(self):
        """
        Builds features from the whole directory and window for each keyword
        :param directory: name of the directory
        :param labelmap: map from string-labels into int-labels
        :param window_st: relative to the phrase position of windows' start (if -5, the windows will contain 5 words before the phrase)
        :param window_end: relative to the phrase position of windows' end
        :return: builded dataset, list of corresponding tuples (raw_filename, keyword) and list of corresponding windows
        """
        if self.labelmap is None:
            labelmap = {'T': 0, 'M': 1, 'O': 2}

        data = []
        kwlist = []
        windows = []

        dir_re = path.join('{}', '*.key').format(self.directory)
        filenames = glob.glob(dir_re)
        for filename in filenames:
            directory, file = raw_filename(filename)
            filename = path.join(directory, file)
            file_data, file_kws = get_data_byfile(filename, self.labelmap)
            data += file_data
            kwlist += file_kws

        self.data = np.array(data)
        self.keywords = kwlist

def get_data_byfile(filename, labelmap):
    """
    build features and word windows for one article
    :param filename: name of the keywords and article file
    :param labelmap: map from string-labels into int-labels
    :param wind_st: relative to the phrase position of windows' start (if -5, the windows will contain 5 words before the phrase)
    :param wind_end: relative to the phrase position of windows' end
    :return: builded dataset, list of corresponding tuples (raw_filename, keyword) and list of corresponding windows
    """
    data = []
    kwlist = []

    keyfile = '{}.key'.format(filename)
    txtfile = '{}.txt'.format(filename)

    kws = get_kwlist_byfile(keyfile, labelmap)

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


    return data, kwlist

def get_kwlist_byfile(filename, labelmap):
    """
    Returns a list of keywords fron one .key file
    :param filename: name of the file
    :param labelmap: map from string-labels into int-labels
    :return: list of tuples (raw_filename, keyword, label)
    """
    data = []
    keywords_mapped = get_keywords(filename, labelmap)
    for label in keywords_mapped:
        for kw in keywords_mapped[label]:
            data.append((raw_filename(filename)[1], kw, label))
    return data

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

    num = wordcount(kw)
    tyend = False
    words = kw.split()
    for word in words:
        if word.endswith('ty'):
            tyend = True

    hasalg = 'algorithm' in kw

    return num, tyend, hasalg

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
