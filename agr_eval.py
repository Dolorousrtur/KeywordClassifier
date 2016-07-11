import glob
import numpy as np
from features import getKeywords
import agreement


def kw_labels_list(directory, labelmap):
    """
    returns a list of keywords and a list of corresponding labels from a given directory
    :param directory: string, the name of a directory where marked keywords are located
    :param labelmap: dict, a map of string-labels and theid corresponding int-labels
    """
    keywords = []
    labels = []
    s = '{}/*.key'.format(directory)
    filenames = glob.glob(s)
    for filename in filenames:
        kwords = getKeywords(filename, labelmap)
        for label in kwords:
            for kw in kwords[label]:
                ind = filename.index('/')
                keywords.append([filename[ind + 1:], kw])
                labels.append(label)
    return keywords, labels


def combined_markings(labelmap, directories):
    """
    combines keywords' labels from several
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

def getAgreement(labels, directories, metric='alpha'):
    """
    returns agreement metric on a given keyword x coder matrix
    :param labels: matrix
    :param directories: tuple of directories with marked keywords NOTE: marking-files (.key) and keywords
    should be the same in all directories
    :param metric: metric to calculate ('kappa','alpha','pi','S')
    :return:
    """
    kws,ls = combined_markings(labels, directories)
    data = agreement.DataSet(agreement.binary_distance)
    vals = []
    for item in range(ls.shape[0]):
        for coder in range(ls.shape[1]):
            vals.append("%s_%s_%s"%(coder,item, ls[item, coder]))
    data.load_array(vals,':')
    return data.get(metric)
