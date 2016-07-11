import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold


def cm2accuracy(cm):
    acc = float(sum(cm[i, i] for i in range(cm.shape[0])))
    return acc / cm.sum()


def kfoldCM(classifier, data, labels, n_classes=None, n_folds=3):
    """
    performs cross-validation on a given dataset with a given classifier
    :param classifier: classifier
    :param data: dataset
    :param labels: labels for the dataset
    :param n_classes: number of classe (if none, determined automatically)
    :param n_folds: number of folds
    :return: confusion matrix for the whole dataset
    """
    kf = KFold(data.shape[0], n_folds=n_folds)
    classes = set(labels)
    if n_classes is None:
        n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes))

    for train_inds, test_inds in kf:
        train_data = data[train_inds, :]
        train_labels = labels[train_inds]
        test_data = data[test_inds, :]
        test_labels = labels[test_inds]

        classifier.fit(train_data, train_labels)
        preds = classifier.predict(test_data)
        cm += confusion_matrix(test_labels, preds, labels=list(classes))
    return cm