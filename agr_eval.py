import glob
from os import path
import numpy as np
from features import get_keywords
from utils_common import raw_filename
import agreement


def kw_labels_list(directory, labelmap):
    """
    returns a list of keywords and a list of corresponding labels from a given directory
    :param directory: string, the name of a directory where marked keywords are located
    :param labelmap: dict, a map of string-labels and their corresponding int-labels
    {'T':0, 'M':1, 'O':2}
    :return list of tuples (article_file, keyword) and a list of corresponding labels
    """
    keywords = []
    labels = []
    s = path.join('{}','*.key').format(directory)
    filenames = glob.glob(s)
    for filename in filenames:
        kwords = get_keywords(filename, labelmap)
        for label in kwords:
            for kw in kwords[label]:
                filename = raw_filename(filename)[1]
                keywords.append((filename, kw))
                labels.append(label)
    return keywords, labels


def combined_markings(labelmap, directories):
    """
    combines keywords' labels from several directories
    :param labelmap: dict, a map of string-labels and theid corresponding int-labels
    :param directories: tuple of directories with marked keywords NOTE: marking-files (.key) and keywords
    should be the same in all directories
    :return: list of keywords (str), combined matrix of labels (np.matrix(N_keywords, N_coders))
    """
    kw_0, l_0 = kw_labels_list(directories[0], labelmap)
    labeltab = np.array(l_0)
    for i in range(1, len(directories)):
        kws, ls = kw_labels_list(directories[i], labelmap)
        ls = np.array([ls[kws.index(w)] for w in kw_0])
        labeltab = np.column_stack((labeltab, ls))
    return kw_0, labeltab


def mismarked_kws(cl_0, cl_1, kwlist, labels_0, labels_1):
    """
    given a list of keywords and two markings returns a list of keywords
    marked by 1st codes as cl_0 and marked by 2nd codes as cl_1
    :param kwlist: list of keywords
    :param labels_0: 1st coder marking
    :param labels_1: 2nd coder marking
    :return:
    """
    kws = [kwlist[i] for i in range(len(kwlist)) if labels_0[i] == cl_0 and labels_1[i] == cl_1]
    return kws


def agreement_bw_directories(labelmap, directories, metric='alpha'):
    """
    Agreement metric for keywords' markings in different directories
    :param labelmap: dict, a map of string-labels and theid corresponding int-labels
    :param directories: tuple of directories with marked keywords NOTE: marking-files (.key) and keywords
    should be the same in all directories
    :param metric: metric to calculate ('kappa','alpha','pi','S')
    :return: metric value
    """
    kws, ls = combined_markings(labelmap, directories)
    return get_agreement(ls)

def get_agreement(labeltab, metric='alpha'):
    """
    returns agreement metric on a given keyword x coder matrix
    :param labels: matrix of kw labels np.array(N_keywords x N_coders)
    :param directories: tuple of directories with marked keywords NOTE: marking-files (.key) and keywords
    should be the same in all directories
    :param metric: metric to calculate ('kappa','alpha','pi','S')
    :return: metric value
    """
    data = agreement.DataSet(agreement.binary_distance)
    vals = []
    for item in range(labeltab.shape[0]):
        for coder in range(labeltab.shape[1]):
            vals.append("%s_%s_%s"%(coder,item, labeltab[item, coder]))
    data.load_array(vals,':')
    return data.get(metric)


