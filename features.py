import os, re
import glob
import numpy as np
from gensim.models import Word2Vec
from nltk import sent_tokenize, word_tokenize

def getTitleAbstractBody(filename):
    """
    retrieve text of title, abstract and body of a given article
    :param filename: article filename
    :return: three strings in lower case: title, abstract, body
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
        title = ""
        abstract = ""
        body = ""
        i = 0
        while lines[i] != "--T\n":
            i += 1
        i += 1
        while lines[i] != "--A\n":
            title += lines[i]
            i += 1
        i += 1
        while lines[i] != "--B\n":
            abstract += lines[i]
            i += 1
        i += 1
        while lines[i] != "--R\n":
            body += lines[i]
            i += 1
        return title.replace('\n', ' ').lower(), abstract.replace('\n', ' ').lower(), body.replace('\n', ' ').lower()

def contains(list, sublist):
    if sublist[0] not in list:
        return False
    else:
        indices = [i for i, x in enumerate(list) if x == sublist[0]]
        for ind in indices:
            occ = True
            for i in range(len(sublist) - 1):
                if list[ind+1+i] != sublist[i+1]:
                    occ = False
                    break
            if occ:
                return (ind, ind+len(sublist))
    return False

def getWindow(text, keyword, fr, to):
    sents = sent_tokenize(text)
    sents_tok = map(lambda x: filter(lambda y: any(c.isalpha() for c in y), word_tokenize(x)), sents)

    window = []
    kw_splitted = keyword.split()

    for sent in sents_tok:
        bounds = contains(sent, kw_splitted)
        if bounds:
            window_st = max(0, bounds[0]+fr)
            window_end = min(len(sent), bounds[1]+to)
            for i in range(window_st, window_end):
                if i not in range(bounds[0], bounds[1]):
                    window.append(sent[i])
    return window

def kwrecDistance(words, target,model):
    size = len(words)
    similarities = []
    for w in words:
        if w in model:
            similarities.append(model.similarity(w, target))
        else:
            similarities.append(0)
    if size == 1:
        return similarities
    else:
        for i in range(size-1):
            combined = '_'.join(words[i:i+2])
            if combined in model:
                words[i:i+2] = (combined,)
                return kwrecDistance(words, target, model)
    return similarities

def keywordDistance(keyword, target, model):
    keyword_parts = keyword.split()
    sim = np.array(kwrecDistance(keyword_parts, target, model)).mean()
    return sim

def text_features(filename, keyword):
    """
    counts keyword's occurences in title, abstract, title and body
    :param filename: article filename
    :param keyword:
    :return: three integers: number of occurences in title, abstract and body
    """
    title, abstract, body = getTitleAbstractBody(filename)
    tcont = title.count(keyword)
    acont = abstract.count(keyword)
    bcont = body.count(keyword)
    if bcont > 0:
        pos = body.index(keyword)
        firstoc = float(len(re.findall('\ +', body[:pos])))/len(re.findall('\ +', body))
    else:
        firstoc = -1
    return tcont, acont, firstoc


def dictListAppend(dict, key, val):
    """
    appends element to a list-value in a dictionary (adds the list and its key to the dictionary in case it was not there)
    :param dict: dictionary
    :param key: key in the dictionary
    :param val: value to append
    """
    if key not in dict:
        dict[key] = []
    dict[key].append(val)

def numWords(keyword):
    return len(keyword.split())

def getKeywords(filename, labels=None):
    """
    generates labelled dictionary of keywords placed by label
    :param filename: file with keywords
    :param labels: map from string-labels into int-labels
    :return: dictionary as {0: [kw0, kw1], 1: [], 2:[], 3:[]}
    """
    keywords = {}

    if labels is None:
        labels = {}

    with open(filename, 'r') as f:
        lines = map(lambda x: x.rstrip(), f.readlines())
        it = iter(lines)
        n = next(it, None)

        maxLabel = 0

        while n is not None:
            while n is not None and n.startswith('-'):
                chlab = n[1:]
                if chlab not in labels:
                    labels[chlab] = maxLabel
                    maxLabel += 1
                currLabel = labels[chlab]
                n = next(it, None)

            while n is not None and not n.startswith('-'):
                if n != '':
                    dictListAppend(keywords, currLabel, n)
                n = next(it, None)

    return keywords


def rawFilename(filename):
    """
    returns filename without extentions
    """
    return os.path.splitext(filename)[0]


def buildData(directory):
    data = []
    labels = {'T': 0, 'M': 1, 'A': 2, 'O': 3}
    kwlist = []
    s = '{}/*.key'.format(directory)
    filenames = glob.glob(s)
    for filename in filenames:
        kwords = getKeywords(filename, labels)

        rawFname = rawFilename(filename)
        for label in kwords:
            for kw in kwords[label]:
                row = np.array(text_features(rawFname + '.txt', kw) + (numWords(kw), ) + (label,))
                data.append(row)
                kwlist.append(kw)
    return np.array(data), kwlist

def countWords(kwlist):
    wordDict = {}
    for kw in kwlist:
        words = kw.lower().split()
        for w in words:
            if w not in wordDict:
                wordDict[w] = 1
            else: wordDict[w] += 1
    return wordDict